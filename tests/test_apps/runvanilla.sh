#!/bin/bash

LD_PRELOAD=../../vanilla/output/lib64/libcuda_hook.so \
REMOTING_BOTTOM_LIBRARY=../../vanilla/output/lib64/libcuda_hook.so \
CUDA_VISIBLE_DEVICES=1 \
python3 $1 $2 $3
