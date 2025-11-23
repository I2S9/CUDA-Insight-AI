#include <cuda_runtime.h>
#include <stdio.h>

// SAXPY kernel: y = a * x + y
// Single-precision A times X Plus Y
__global__ void saxpy(float a, float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

// Host wrapper function
extern "C" void launch_saxpy(float a, float* d_x, float* d_y, int n, int threads_per_block) {
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    saxpy<<<blocks, threads_per_block>>>(a, d_x, d_y, n);
    cudaDeviceSynchronize();
}

// Simple test function that allocates memory and runs the kernel
extern "C" int test_saxpy() {
    const int n = 1024;
    const float a = 2.0f;
    const int threads_per_block = 256;
    
    float *h_x = new float[n];
    float *h_y = new float[n];
    float *d_x, *d_y;
    
    // Initialize host data
    for (int i = 0; i < n; i++) {
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
    }
    
    // Allocate device memory
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));
    
    // Copy to device
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    saxpy<<<blocks, threads_per_block>>>(a, d_x, d_y, n);
    cudaDeviceSynchronize();
    
    // Copy back
    cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_x);
    cudaFree(d_y);
    delete[] h_x;
    delete[] h_y;
    
    return 0;
}

