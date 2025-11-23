#include <cuda_runtime.h>
#include <stdio.h>

// Vector addition kernel: c = a + b
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Host wrapper function
extern "C" void launch_vector_add(float* d_a, float* d_b, float* d_c, int n, int threads_per_block) {
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    vector_add<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
}

// Simple test function that allocates memory and runs the kernel
extern "C" int test_vector_add() {
    const int n = 1024;
    const int threads_per_block = 256;
    
    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_c = new float[n];
    float *d_a, *d_b, *d_c;
    
    // Initialize host data
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    // Allocate device memory
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));
    
    // Copy to device
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    vector_add<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    
    // Copy back
    cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    
    return 0;
}

