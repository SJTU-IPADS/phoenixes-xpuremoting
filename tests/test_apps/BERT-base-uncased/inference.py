import torch
from transformers import pipeline
import time
import sys
import ctypes
import os

# load remoting bottom library
path = os.getenv('REMOTING_BOTTOM_LIBRARY')
cpp_lib = ctypes.CDLL(path)
start_trace = cpp_lib.startTrace

if(len(sys.argv) != 2):
    print('Usage: python3 inference.py num_iter')
    sys.exit()

num_iter = int(sys.argv[1])

# get model
model = 'bert-base-uncased'
unmasker = pipeline('fill-mask', model = model, device = 0)

# remove initial overhead
for i in range(20):
    result = unmasker("Hello I'm a [MASK] model.")

start_trace()

T1 = time.time()

for i in range(num_iter):
    result = unmasker("Hello I'm a [MASK] model.")
    
T2 = time.time()
print('time used: ', T2-T1)
print(result)
