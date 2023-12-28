#!/bin/bash

CUDA_VISIBLE_DEVICES=1 \
LD_LIBRARY_PATH=../../submodules/libtirpc/install/lib \
XPU_REMOTE_ADDRESS=$XPU_REMOTE_ADDRESS \
../../cpu/cricket-rpc-server