#!/bin/bash

# for python
LD_LIBRARY_PATH=./submodules/libtirpc/install/lib:./cpu/ LD_PRELOAD=./cpu/cricket-client.so REMOTE_GPU_ADDRESS=127.0.0.1 python3 $1

# for debug python
# LD_LIBRARY_PATH=./submodules/libtirpc/install/lib:./cpu/ LD_PRELOAD=./cpu/cricket-client.so REMOTE_GPU_ADDRESS=127.0.0.1 gdb -ex r --args python3 $1

# for cuda
# LD_LIBRARY_PATH=./submodules/libtirpc/install/lib:./cpu/ LD_PRELOAD=./cpu/cricket-client.so REMOTE_GPU_ADDRESS=127.0.0.1 gdb -ex r --args $1