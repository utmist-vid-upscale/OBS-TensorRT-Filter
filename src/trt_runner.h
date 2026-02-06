#ifndef TRT_RUNNER_H
#define TRT_RUNNER_H

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// TensorRT runner context
struct trt_runner {
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    cudaStream_t stream;
    
    // Tensor names
    const char* input_name;
    const char* output_name;
    
    // Tensor info
    nvinfer1::Dims input_dims;
    nvinfer1::Dims output_dims;
    nvinfer1::DataType input_type;
    nvinfer1::DataType output_type;
    
    // Buffer sizes
    size_t input_bytes;
    size_t output_bytes;
    
    // Device pointers (must be set before inference)
    void* d_input;
    void* d_output;
    
    bool initialized;
};

// Initialize TensorRT runner from engine file
// Returns true on success, false on failure
bool trt_runner_init(struct trt_runner* runner, const char* engine_path, cudaStream_t stream);

// Set input/output device pointers (must be valid CUDA device memory)
void trt_runner_set_buffers(struct trt_runner* runner, void* d_input, void* d_output);

// Run inference (async, returns immediately)
// Returns true on success, false on failure
bool trt_runner_enqueue(struct trt_runner* runner);

// Cleanup resources
void trt_runner_destroy(struct trt_runner* runner);

#ifdef __cplusplus
}
#endif

#endif // TRT_RUNNER_H
