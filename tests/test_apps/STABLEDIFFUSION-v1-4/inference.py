from diffusers import StableDiffusionPipeline
import time
import sys

if(len(sys.argv) != 2):
    print('Usage: python3 inference.py num_iter')
    sys.exit()

num_iter = int(sys.argv[1])

model = "CompVis/stable-diffusion-v1-4"
image_pipe = StableDiffusionPipeline.from_pretrained(model)

device = "cuda"
image_pipe = image_pipe.to(device)
prompt = "a photograph of an astronaut riding a horse"

# remove initial overhead
for i in range(20):
    out_images = image_pipe(prompt).images

T1 = time.time()

for i in range(num_iter):
    out_images = image_pipe(prompt).images

T2 = time.time()
print('time used: ', T2-T1)
out_images[0].save("astronaut_rides_horse.png")
