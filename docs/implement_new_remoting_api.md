## 0. Pre-requests

1. 确保MPS关闭：

TBD：instructions on how to check whether MPS is enabled (and instructions to disable it) 

2. Prepare docker container

## 1. Build 

1. Clone the repo:

```
git clone --recursive -b share-object-support  git@ipads.se.sjtu.edu.cn:scaleaisys/cricket.git
```

2. 启动所需的容器

假设cricket被clone到了 $CRICKET

```
docker run --gpus all -dit -v $CRICKET:/testdir --privileged --network host --ipc=host --namecricket-pytorch-xxx yyh/pytorch:v1.13.1-devel-new
```

3. 编译：

```
docker exec -it cricket-pytorch-xxx bash
cd /testdir
make libtirpc
cd cpu && make cricket-rpc-server cricket-client.so
```

## 2. Run 

1. 运行cricket server

```
docker exec -it cricket-pytorch-xxx bash
cd /testdir/cpu
LD_LIBRARY_PATH=../submodules/libtirpc/install/lib ./cricket-rpc-server
```

2. 运行client（真实的python程序）

```
docker exec -it cricket-pytorch-xxx bash
cd /testdir
LD_LIBRARY_PATH=./submodules/libtirpc/install/lib LD_PRELOAD=./cpu/cricket-client.so REMOTE_GPU_ADDRESS=127.0.0.1 python3 ./tests/test_apps/pytorch_minimal.py
```

这个程序会执行大约3-4分钟才能完成。client会正常退出，server则不会退出。

## 3. Run an example of unsupported code

1. example cuda code

```c
#include <cuda_runtime.h>
#include <iostream>

int main() {
    int size = 1024 * sizeof(int);
    int* devicePtr;

    cudaMalloc((void**)&devicePtr, size);

    cudaMemsetAsync(devicePtr, 0, size);

    cudaDeviceSynchronize();

    cudaFree(devicePtr);

    return 0;
}
```

将该代码存在tests/test_apps/cuda_memset_async.cu中。

通过该命令编译

```bash
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
LD_LIBRARY_PATH=./submodules/libtirpc/install/lib LD_PRELOAD=./cpu/cricket-client.so REMOTE_GPU_ADDRESS=127.0.0.1 ./tests/test_apps/a.out
```

3. 未支持api的表现

client的输出

```
+00:00:00.000006 INFO:  connection to host "127.0.0.1"
+00:00:00.000130 DEBUG: the command is "a.out"
+00:00:00.000147 DEBUG: using prog=99, vers=1   in cpu-client.c:98
+00:00:00.000163 INFO:  connecting via TCP...
+00:00:00.001235 DEBUG: __cudaRegisterFatBinary(fatCubin=0x561bc7f52028)      in cpu-client.c:360
00000: 7f454c46 02010133 07000000 00000000  | .ELF...3........
+00:00:00.001477 WARNING: could not find .nv.info section. This means this binary does not contain any kernels.        in cpu-elf2.c:924
+00:00:00.001905 DEBUG: fatbin loaded to 0x561bc87dab50
Segmentation fault (core dumped)
```

使用gdb的话，栈回溯是类似这样

```
Program received signal SIGSEGV, Segmentation fault.
0x0000000000000000 in ?? ()
(gdb) bt
#0  0x0000000000000000 in ?? ()
#1  0x00007ffff7b6b566 in dlopen () from ./cpu/cricket-client.so
#2  0x00007ffff7b8ed3d in libwrap_get_sohandle () from ./cpu/cricket-client.so
#3  0x00007ffff7b75805 in cudaMemsetAsync () from ./cpu/cricket-client.so
#4  0x0000555555554b65 in main ()
```

有时，部分API也会是死递归的表现（打了很多很多log，然后segfault，爆栈了）

实现这个cudaMemsetAsync的方法可以参考这个[commit](https://ipads.se.sjtu.edu.cn:1312/scaleaisys/cricket/-/commit/4dc9a9d39db6b996d2c01cb07d0285a58a73298f)，把这个commit反着做就可以了。

## 4. 如何实现新的API

参考这个[文档](https://ipads.se.sjtu.edu.cn:1312/scaleaisys/cricket/-/blob/share-object-support/docs/how_to_add_support_cuda_calls.md)
