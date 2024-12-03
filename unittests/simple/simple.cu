#include <stdio.h>
#include <cuda_runtime.h>

// A device function that adds two arrays and stores the result in a third array
__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// A host function that adds two arrays and stores the result in a third array
void addKernelHost(int *c, const int *a, const int *b, int i)
{
    c[i] = a[i] + b[i];
}

int main()
{
    printf("in the main function of cuda sample\n");
    printf("function pointer of addKernel: %p\n", addKernel);
    printf("function pointer of addKernelHost: %p\n", addKernelHost);
    // Declare host arrays
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    printf("pointer to the stack variable: %p\n", &arraySize);

    // Declare device pointers
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;

    // Allocate device memory
    cudaMalloc((void**)&dev_c, arraySize * sizeof(int));
    cudaMalloc((void**)&dev_a, arraySize * sizeof(int));
    cudaMalloc((void**)&dev_b, arraySize * sizeof(int));

    // Copy host arrays to device memory
    cudaMemcpy(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with one block and five threads
    void* args[] = { &dev_c, &dev_a, &dev_b };
    cudaLaunchKernel((void*)addKernel, dim3(1), dim3(arraySize), args);

    // Copy device array to host memory
    cudaMemcpy(c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    printf("{%d, %d, %d, %d, %d} + {%d, %d, %d, %d, %d} = {%d, %d, %d, %d, %d}\n",
        a[0], a[1], a[2], a[3], a[4],
        b[0], b[1], b[2], b[3], b[4],
        c[0], c[1], c[2], c[3], c[4]);

    // Free device memory
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return 0;
}
