#include <cuda_runtime.h>
#include <stdio.h>

int main()
{
    const int iterations = 10000;
    const int arraySize = 250000;
    int a[arraySize] = {}, b[arraySize] = {};

    int *dev_a = nullptr;
    cudaMalloc((void **)&dev_a, arraySize * sizeof(int));

    for (int i = 0; i < arraySize; i++) {
        a[i] = i;
    }

    for (int i = 0; i < iterations; i++) {
        // printf("b[0] = %d\n", b[0]);
        cudaMemcpy(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(b, dev_a, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
        // printf("b[arraySize / 2] = %d\n", b[arraySize / 2]);
    }

    cudaFree(dev_a);

    return 0;
}
