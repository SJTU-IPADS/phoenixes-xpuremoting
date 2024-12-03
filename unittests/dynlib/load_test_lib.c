#include <stdio.h>
#include <dlfcn.h>
#include <assert.h>
#include <cuda_runtime.h>

typedef cudaError_t (*kernel_type)(int *, const int *, const int *, int);
typedef cudaError_t (*cuda_malloc_type)(void**, size_t);
typedef cudaError_t (*cuda_memcpy_type)(void*, const void*, size_t, enum cudaMemcpyKind);
typedef cudaError_t (*cuda_free_type)(void*);

#define arraySize (5)

int main() {
    void* handler = dlopen("./test_lib.so", RTLD_LAZY);
    if (!handler) {
        printf("Error: %s\n", dlerror());
        return 1;
    }
    assert(handler);
    kernel_type func = (kernel_type) dlsym(handler, "addKernelHost");
    cuda_malloc_type cuda_malloc_wrapper = (cuda_malloc_type) dlsym(handler, "cudaMallocWrapper");
    cuda_memcpy_type cuda_memcpy_wrapper = (cuda_memcpy_type) dlsym(handler, "cudaMemcpyWrapper");
    cuda_free_type cuda_free_wrapper = (cuda_free_type) dlsym(handler, "cudaFreeWrapper");
    assert(func);
    assert(cuda_malloc_wrapper);
    assert(cuda_memcpy_wrapper);
    assert(cuda_free_wrapper);

    // Declare host arrays
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Declare device pointers
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;

    // Allocate device memory
    cuda_malloc_wrapper((void**)&dev_c, arraySize * sizeof(int));
    cuda_malloc_wrapper((void**)&dev_a, arraySize * sizeof(int));
    cuda_malloc_wrapper((void**)&dev_b, arraySize * sizeof(int));

    // Copy host arrays to device memory
    cuda_memcpy_wrapper(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cuda_memcpy_wrapper(dev_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    func(dev_c, dev_a, dev_b, arraySize);

    // Copy device array to host memory
    cuda_memcpy_wrapper(c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    printf("{%d, %d, %d, %d, %d} + {%d, %d, %d, %d, %d} = {%d, %d, %d, %d, %d}\n",
        a[0], a[1], a[2], a[3], a[4],
        b[0], b[1], b[2], b[3], b[4],
        c[0], c[1], c[2], c[3], c[4]);
    
    // Free device memory
    cuda_free_wrapper(dev_c);
    cuda_free_wrapper(dev_a);
    cuda_free_wrapper(dev_b);
    return 0;
}
