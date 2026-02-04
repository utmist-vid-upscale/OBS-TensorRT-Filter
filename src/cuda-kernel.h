#ifndef CUDA_KERNEL_H
#define CUDA_KERNEL_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Launch the invert BGRA kernel
// d_input: device pointer to input BGRA image
// d_output: device pointer to output BGRA image
// width: image width in pixels
// height: image height in pixels
// stream: CUDA stream to launch kernel on
cudaError_t launch_invert_kernel(
    unsigned char *d_input,
    unsigned char *d_output,
    int width,
    int height,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // CUDA_KERNEL_H
