# Remote Microbench

## 0. Microbench Running

- run a docker container mounting the cricket directory and enter it. If running on meepo4, can use image "hjb/pytorch:v1.13.1-devel".

- `cd path/to/cricket`

- `make cpu`

- `cd path/to/microbench`

- `./startserver.sh` in one shell, and `./startclient.sh ./cuda_app/xxx.out` in the another.

## 1. Measured APIs

### 1.1. Control Path

- CUDA_GET_DEVICE
- __cudaRegisterFatBinary & __cudaRegisterFunction
- cudnnCreateTensorDescriptor & cudnnDestroyTensorDescriptor

### 1.2. Data Path

- CUDA_MEMCPY
- CUDA_LAUNCH_KERNEL

### 1.3. Call Stack of CUDA API

```plain
#0 cudaSetDevice(int device), at cpu-client-runtime.c (intercepting the cuda executable binary)

#1 cuda_set_device_1(int arg1, int *clnt_res, CLIENT *clnt), at cpu_rpc_prot_clnt.c

#2 clnt_vc_call(cl, proc, xdr_args, args_ptr, xdr_results, results_ptr, timeout), at clnt_vc.c
```

前两层是参数直接的转发（内存拷贝），RPC 处理过程在 clnt_vc_call 中，

- `AUTH_WRAP(cl->cl_auth, xdrs, xdr_args, args_ptr)` 等进行序列化。 **serialization time**
- 底层的 `flush_out(rstrm, eor)` 和 `fill_input_buf(rstrm)` 通过调用 `write/read(tcp_fd, ...)` 将读写缓冲区数据传输。 **network time**
- `AUTH_UNWRAP(cl->cl_auth, xdrs, xdr_results, results_ptr)` 等进行反序列化。 **serialization time**

在 server 端的 `cuda_get_device_1_svc(int_result *result, struct svc_req *rqstp)` 中实际运行 cuda api。 **vanilla time**

剩余时间记为 **memcpy time**。

## 2. Questions

### 2.1. mem_data XDR method

``` plain
bool_t
xdr_mem_data (XDR *xdrs, mem_data *objp)
{
	register int32_t *buf;
	printf("xdr_pos = %d, ", XDR_GETPOS(xdrs));
	 if (!xdr_bytes (xdrs, (char **)&objp->mem_data_val, (u_int *) &objp->mem_data_len, ~0))
		 return FALSE;
	printf("mem_data_len = %d, xdr_pos = %d\n", objp->mem_data_len, XDR_GETPOS(xdrs));
	return TRUE;
}
```

实际 memory_size = 1000000，但 xdr pos 前后变化 17080。

**原因：**

XDR 在 xdr_rec.c 中定义为一个流式 I/O 缓冲区，read/write buffer 大小都限制为 65536，达到限制会立即 flush 掉。所以实际上 mem_data 传输被分成了很多次的 `write(tcp_fd, ...)`。
