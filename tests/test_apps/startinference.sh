#!/bin/bash

LD_LIBRARY_PATH=../../submodules/libtirpc/install/lib:../../cpu/ \
LD_PRELOAD=../../cpu/cricket-client.so \
REMOTING_BOTTOM_LIBRARY=../../cpu/cricket-client.so \
REMOTE_GPU_ADDRESS=127.0.0.1 \
python3 $1 $2 $3
