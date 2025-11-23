#include <cuda_runtime.h>
#include <stdio.h>

// Example kernel with potential warp divergence
// This kernel processes even and odd indices differently
__global__ void divergent_kernel(float* input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // This condition can cause warp divergence
    // because threads in the same warp take different paths
    if (i % 2 == 0) {
        output[i] = input[i] * 2.0f;
    } else {
        output[i] = input[i] / 2.0f;
    }
}

