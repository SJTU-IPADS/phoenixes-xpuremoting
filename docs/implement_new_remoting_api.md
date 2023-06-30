## 0. Pre-requests

1. 确保MPS关闭：

TBD：instructions on how to check whether MPS is enabled (and instructions to disable it) 

2. Prepare docker container



## 1. Build 

1. Clone the repo:

```
git clone --recursive -b share-object-support  git@ipads.se.sjtu.edu.cn:scaleaisys/cricket.git
```

2. 启动容器编译所需的容器

假设criket被clone到了 $CRIKET$ 

```
docker run --gpus all -dit -v $CRIKET$:/testdir --privileged --network host --ipc=host --namecricket-pytorch-xxx yyh/pytorch:v1.13.1-devel-new
```

3. 编译：

```
docker exec -it cricket-pytorch-xxx bash
cd /testdir
make libtirpc
cd cpu && make cricket-rpc-server cricket-client.so
```



## 2. Run 

1. 运行criket server

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

