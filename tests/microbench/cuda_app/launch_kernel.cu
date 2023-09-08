#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void addKernel(int *c, const int *a, const int *b, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char **argv)
{
    const int size = 1000000;
    const int iterations = 10000;
    int a[size] = { 0 };
    int *dev_a = nullptr;

    // Allocate GPU buffers for three vectors (two input, one output)
    cudaMalloc((void **)&dev_a, size * sizeof(int));

    // Copy input vectors from host memory to GPU buffers.
    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch a kernel on the GPU with one thread for each element.
    for (int i = 0; i < iterations; i++) {
        addKernel<<<2, (size + 1) / 2>>>(dev_a, dev_a, dev_a, size);
        cudaDeviceSynchronize();
    }

    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(a, dev_a, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_a);

    return 0;
}
