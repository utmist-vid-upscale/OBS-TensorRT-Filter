#include "trt_runner.h"
#include <obs-module.h>
#include <fstream>
#include <vector>
#include <cstring>

// TensorRT logger
class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            blog(LOG_WARNING, "[TensorRT] %s", msg);
        } else {
            blog(LOG_DEBUG, "[TensorRT] %s", msg);
        }
    }
};

static TRTLogger g_trt_logger;

// Load engine file
static std::vector<char> load_engine_file(const char* engine_path) {
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        blog(LOG_ERROR, "[TensorRT] Failed to open engine file: %s", engine_path);
        return std::vector<char>();
    }
    
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        blog(LOG_ERROR, "[TensorRT] Failed to read engine file");
        return std::vector<char>();
    }
    
    blog(LOG_INFO, "[TensorRT] Loaded engine file: %s (%zu bytes)", engine_path, size);
    return buffer;
}

bool trt_runner_init(struct trt_runner* runner, const char* engine_path, cudaStream_t stream) {
    memset(runner, 0, sizeof(*runner));
    runner->stream = stream;
    
    // Load engine file
    std::vector<char> engine_data = load_engine_file(engine_path);
    if (engine_data.empty()) {
        return false;
    }
    
    // Create TensorRT runtime
    runner->runtime = nvinfer1::createInferRuntime(g_trt_logger);
    if (!runner->runtime) {
        blog(LOG_ERROR, "[TensorRT] Failed to create runtime");
        return false;
    }
    
    // Deserialize engine
    runner->engine = runner->runtime->deserializeCudaEngine(
        engine_data.data(),
        engine_data.size()
    );
    
    if (!runner->engine) {
        blog(LOG_ERROR, "[TensorRT] Failed to deserialize engine");
        delete runner->runtime;
        runner->runtime = nullptr;
        return false;
    }
    
    blog(LOG_INFO, "[TensorRT] Engine deserialized successfully");
    
    // Get tensor names
    int num_io_tensors = runner->engine->getNbIOTensors();
    blog(LOG_INFO, "[TensorRT] Number of IO tensors: %d", num_io_tensors);
    
    for (int i = 0; i < num_io_tensors; i++) {
        const char* tensor_name = runner->engine->getIOTensorName(i);
        bool is_input = runner->engine->getTensorIOMode(tensor_name) == nvinfer1::TensorIOMode::kINPUT;
        
        if (is_input) {
            runner->input_name = tensor_name;
            blog(LOG_INFO, "[TensorRT] Input tensor: %s", tensor_name);
        } else {
            runner->output_name = tensor_name;
            blog(LOG_INFO, "[TensorRT] Output tensor: %s", tensor_name);
        }
    }
    
    if (!runner->input_name || !runner->output_name) {
        blog(LOG_ERROR, "[TensorRT] Failed to find input or output tensor");
        delete runner->engine;
        delete runner->runtime;
        runner->engine = nullptr;
        runner->runtime = nullptr;
        return false;
    }
    
    // Get tensor shapes and types
    runner->input_dims = runner->engine->getTensorShape(runner->input_name);
    runner->output_dims = runner->engine->getTensorShape(runner->output_name);
    runner->input_type = runner->engine->getTensorDataType(runner->input_name);
    runner->output_type = runner->engine->getTensorDataType(runner->output_name);
    
    // Calculate sizes
    size_t input_size = 1;
    for (int i = 0; i < runner->input_dims.nbDims; i++) {
        input_size *= runner->input_dims.d[i];
    }
    
    size_t output_size = 1;
    for (int i = 0; i < runner->output_dims.nbDims; i++) {
        output_size *= runner->output_dims.d[i];
    }
    
    size_t input_element_size = (runner->input_type == nvinfer1::DataType::kFLOAT) ? sizeof(float) : sizeof(__half);
    size_t output_element_size = (runner->output_type == nvinfer1::DataType::kFLOAT) ? sizeof(float) : sizeof(__half);
    
    runner->input_bytes = input_size * input_element_size;
    runner->output_bytes = output_size * output_element_size;
    
    blog(LOG_INFO, "[TensorRT] Input: %zu elements (%zu bytes), Output: %zu elements (%zu bytes)",
         input_size, runner->input_bytes, output_size, runner->output_bytes);
    
    // Create execution context
    runner->context = runner->engine->createExecutionContext();
    if (!runner->context) {
        blog(LOG_ERROR, "[TensorRT] Failed to create execution context");
        delete runner->engine;
        delete runner->runtime;
        runner->engine = nullptr;
        runner->runtime = nullptr;
        return false;
    }
    
    runner->initialized = true;
    return true;
}

void trt_runner_set_buffers(struct trt_runner* runner, void* d_input, void* d_output) {
    if (!runner->initialized || !runner->context) {
        return;
    }
    
    runner->d_input = d_input;
    runner->d_output = d_output;
    
    // Set tensor addresses
    runner->context->setTensorAddress(runner->input_name, d_input);
    runner->context->setTensorAddress(runner->output_name, d_output);
}

bool trt_runner_enqueue(struct trt_runner* runner) {
    if (!runner->initialized || !runner->context) {
        blog(LOG_ERROR, "[TensorRT] Runner not initialized");
        return false;
    }
    
    if (!runner->d_input || !runner->d_output) {
        blog(LOG_ERROR, "[TensorRT] Buffers not set");
        return false;
    }
    
    // Run inference
    bool success = runner->context->enqueueV3(runner->stream);
    if (!success) {
        blog(LOG_ERROR, "[TensorRT] Inference failed");
        return false;
    }
    
    return true;
}

void trt_runner_destroy(struct trt_runner* runner) {
    if (!runner) {
        return;
    }
    
    if (runner->context) {
        delete runner->context;
        runner->context = nullptr;
    }
    
    if (runner->engine) {
        delete runner->engine;
        runner->engine = nullptr;
    }
    
    if (runner->runtime) {
        delete runner->runtime;
        runner->runtime = nullptr;
    }
    
    runner->initialized = false;
}
