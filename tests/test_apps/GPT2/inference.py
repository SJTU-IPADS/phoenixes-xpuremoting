import torch
from transformers import pipeline, set_seed
import time
import sys

if(len(sys.argv) != 2):
    print('Usage: python3 inference.py num_iter')
    sys.exit()

num_iter = int(sys.argv[1])

# get model
model = 'gpt2'
generator = pipeline('text-generation', model = model, device = 0)
set_seed(42)

# remove initial overhead
for i in range(20):
    result = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)

T1 = time.time()

for i in range(num_iter):
    result = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
    
T2 = time.time()
print('time used: ', T2-T1)
print(result)
