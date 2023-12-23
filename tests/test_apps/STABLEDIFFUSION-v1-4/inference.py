import torch
from diffusers import StableDiffusionPipeline, StableDiffusionOnnxPipeline
import time
import sys
import ctypes
import os

# load remoting bottom library
path = os.getenv('REMOTING_BOTTOM_LIBRARY')
cpp_lib = ctypes.CDLL(path)
start_trace = cpp_lib.startTrace

if(len(sys.argv) != 3):
    print('Usage: python3 inference.py num_iter batch_size')
    sys.exit()

num_iter = int(sys.argv[1])
batch_size = int(sys.argv[2])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="main",
    torch_dtype=torch.float32,
).to(device)

prompt = "a photo of an astronaut riding a horse on mars"

# if batch_size>16:
#     pipe.enable_vae_slicing()

# remove initial overhead
torch.cuda.empty_cache()
for i in range(2):
    images = pipe(prompt=[prompt] * batch_size, num_inference_steps=50).images

start_trace()

T1 = time.time()

for i in range(num_iter):
    images = pipe(prompt=[prompt] * batch_size, num_inference_steps=50).images
    
T2 = time.time()
print('time used: ', T2-T1)
# images[0].save("astronaut_rides_horse.png")
