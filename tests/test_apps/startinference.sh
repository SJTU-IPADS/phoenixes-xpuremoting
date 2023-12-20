#!/bin/bash

WORK_PATH=$(cd $(dirname $0) && pwd) && export REMOTING_BOTTOM_LIBRARY=$WORK_PATH/../../cpu/cricket-client.so
LD_LIBRARY_PATH=../../submodules/libtirpc/install/lib:../../cpu/ LD_PRELOAD=../../cpu/cricket-client.so REMOTE_GPU_ADDRESS=127.0.0.1 python3 $1 $2
