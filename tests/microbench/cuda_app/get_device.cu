#include <cuda_runtime.h>
#include <stdio.h>

int main()
{
    const int iterations = 10000;
    int device;
    for (int i = 0; i < iterations; ++i) {
        cudaGetDevice(&device);
        // printf("cuda device: %d\n", device);
    }

    return 0;
}
