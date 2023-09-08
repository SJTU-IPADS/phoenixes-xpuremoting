#include <stdio.h>
#include <cuda_runtime.h>

// A device function that adds two arrays and stores the result in a third array
extern "C" __global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

extern "C" void addKernelHost(int *c, const int *a, const int *b, int arraySize)
{
    printf("c: %p, a: %p, b: %p\n", c, a, b);
    addKernel<<<1, arraySize>>>(c, a, b);
}

extern "C" cudaError_t cudaMallocWrapper(void **p, size_t s)
{
    return cudaMalloc(p, s);
}

extern "C" cudaError_t cudaMemcpyWrapper(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
{
    return cudaMemcpy(dst, src, count, kind);
}

extern "C" cudaError_t cudaFreeWrapper(void *devPtr)
{
    return cudaFree(devPtr);
}
