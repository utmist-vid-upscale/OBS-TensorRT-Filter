#include <cuda_runtime.h>

// Simple CUDA kernel to invert BGRA pixels
__global__ void invert_bgra_kernel(
    unsigned char *input,
    unsigned char *output,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height)
        return;
    
    int idx = (y * width + x) * 4; // BGRA = 4 bytes per pixel
    
    // Invert RGB channels, keep alpha
    output[idx + 0] = 255 - input[idx + 0]; // B
    output[idx + 1] = 255 - input[idx + 1]; // G
    output[idx + 2] = 255 - input[idx + 2]; // R
    output[idx + 3] = input[idx + 3];       // A
}

// C-callable wrapper function to launch the kernel
extern "C" cudaError_t launch_invert_kernel(
    unsigned char *d_input,
    unsigned char *d_output,
    int width,
    int height,
    cudaStream_t stream)
{
    // Calculate grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    // Launch kernel
    invert_bgra_kernel<<<gridSize, blockSize, 0, stream>>>(
        d_input, d_output, width, height);
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }
    
    return cudaSuccess;
}
