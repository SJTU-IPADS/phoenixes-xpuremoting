#!/bin/bash

VERSION=WITH_VANILLA INFERENCE=$1 python3 inference_experiment.py $2 $3
VERSION=WITH_SHARED_MEMORY INFERENCE=$1 python3 inference_experiment.py $2 $3
VERSION=WITH_RDMA INFERENCE=$1 python3 inference_experiment.py $2 $3
# VERSION=WITH_TCPIP INFERENCE=$1 python3 inference_experiment.py $2
# VERSION=NO_OPTIMIZATION INFERENCE=$1 python3 inference_experiment.py $2
