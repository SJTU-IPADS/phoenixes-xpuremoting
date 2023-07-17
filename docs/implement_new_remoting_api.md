# Tutorial on implementing CUDA remote invocation 

Credits: Tianxia Wang, Xingda Wei, Wenxin Zheng 


## 0. Pre-requests

1. 确保MPS关闭：

由于用户可以设定`CUDA_MPS_PIPE_DIRECTORY`为非默认目录，因此不能通过判断`/tmp/nvidia-mps`下是否存在MPS_PIPE确定是否关闭mps。

现阶段可以通过`ps -ef | grep nvidia-cuda-mps-control`检查是否存在mps进程。

```
> ps -ef | grep nvidia-cuda-mps-control
zwx       3850  3593  0 19:42 pts/20   00:00:00 nvidia-cuda-mps-control -f
zwx       4589  4035  0 19:42 pts/21   00:00:00 grep --color=auto --exclude-dir=.bzr --exclude-dir=CVS --exclude-dir=.git --exclude-dir=.hg --exclude-dir=.svn --exclude-dir=.idea --exclude-dir=.tox nvidia-cuda-mps-control
```
如果存在这个进程，确认没有CUDA任务负载后，可以通过以下方法关闭mps。

`echo "quit" | sudo nvidia-cuda-mps-control` 或 `pkill nvidia-cuda-mps-control`

2. Prepare docker container, 参考： [prepare_container.md](prepare_container.md) 

## 1. Build 

1. Clone the repo:

```
git clone --recursive -b share-object-support  git@ipads.se.sjtu.edu.cn:scaleaisys/cricket.git
```

2. 启动所需的容器

假设cricket被clone到了 $CRICKET

```
docker run --gpus all -dit -v $CRICKET:/testdir --privileged --network host --ipc=host --name cricket-pytorch-xxx yyh/pytorch:v1.13.1-devel-new
```

3. 编译：

```
docker exec -it cricket-pytorch-xxx bash
cd /testdir
make libtirpc
cd cpu && make cricket-rpc-server cricket-client.so
```

**Note**: 默认编译的debug level是`DEBUG`， 会输出很多无效信息。正常跑可以使用如下选项避免：

```
# In docker container 
cd cpu && LOG=INFO make cricket-rpc-server cricket-client.so
```

## 2. Run 

1. 运行cricket server

```
docker exec -it cricket-pytorch-xxx bash
cd /testdir/cpu
LD_LIBRARY_PATH=../submodules/libtirpc/install/lib ./cricket-rpc-server
```

2. 运行client（使用GPU的程序）

```
docker exec -it cricket-pytorch-xxx bash
cd /testdir

LD_LIBRARY_PATH=./submodules/libtirpc/install/lib:./cpu/cricket-client.so LD_PRELOAD=./cpu/cricket-client.so REMOTE_GPU_ADDRESS=127.0.0.1 python3 ./tests/test_apps/pytorch_minimal.py
```

如果一切正常，会打印如下：

```
1499 9.506717681884766
1599 9.303845405578613
1699 9.160715103149414
1799 9.059717178344727
1899 8.988435745239258
1999 8.938121795654297
Result: y = 0.011471157893538475 + 0.8549202084541321 x + -0.001978963380679488 x^2 + -0.09307142347097397 x^3
+00:01:34.067282 INFO:	api-call-cnt: 512150
+00:01:34.067312 INFO:	memcpy-cnt: 8016
```



**Note**: 一般程序执行会比较长，因为kernel需要动态parse kernel。这些parse的目前存在client的内存中。未来可以采用cache的方案避免每次执行都进行parse。



## 3. Run an example of code with unsupported API

1. Example CUDA code (Note that we use `c++` as an exmaple, but pytorch code runs just fine).  We use the sample code in `tests/test_apps/cuda_memset_async.cu`. 

```c
#include <cuda_runtime.h>
#include <iostream>

int main() {
    int size = 1024 * sizeof(int);
    int* devicePtr;

    cudaMalloc((void**)&devicePtr, size);

    cudaMemsetAsync(devicePtr, 0, size); // <- the API not implemented

    cudaDeviceSynchronize();

    cudaFree(devicePtr);

    return 0;
}
```



通过该命令编译

```bash
## Note in the docker container, e.g., via docker exec -it cricket-pytorch-xxx bash
nvcc tests/test_apps/cuda_memset_async.cu -cudart shared -o tests/test_apps/a.out
```

2. 运行cricket server和cricket client

server

```bash
docker exec -it cricket-pytorch-xxx bash
cd /testdir/cpu
LD_LIBRARY_PATH=../submodules/libtirpc/install/lib ./cricket-rpc-server
```

client

```bash
docker exec -it cricket-pytorch-xxx bash
cd /testdir/cpu
LD_LIBRARY_PATH=./submodules/libtirpc/install/lib:./cpu/cricket-client.so  LD_PRELOAD=./cpu/cricket-client.so REMOTE_GPU_ADDRESS=127.0.0.1 ./tests/test_apps/a.out
```

3. 如果未实现某个API的话，一般会把具体的error print出来：

```
+00:00:00.001808 DEBUG:	fatbin loaded to 0x559754d81b50
+00:00:00.002558 ERROR: un-implemented function cudaMemsetAsync	in cpu-client-runtime.c:1822
a.out: cpu-client-runtime.c:1822: cudaMemsetAsync: Assertion `0' failed.
```

一般会报assert failure，如上所示。

实现这个cudaMemsetAsync的方法可以参考这个[commit](https://ipads.se.sjtu.edu.cn:1312/scaleaisys/cricket/-/commit/4dc9a9d39db6b996d2c01cb07d0285a58a73298f)，把这个commit反着做就可以了。

## 4. 如何实现新的API

参考这个[文档](https://ipads.se.sjtu.edu.cn:1312/scaleaisys/cricket/-/blob/share-object-support/docs/how_to_add_support_cuda_calls.md)
