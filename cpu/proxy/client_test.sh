#!/bin/bash

LD_LIBRARY_PATH=../../submodules/libtirpc/install/lib:../ LD_PRELOAD=../cricket-client.so REMOTE_GPU_ADDRESS=127.0.0.1 ./simple.out
