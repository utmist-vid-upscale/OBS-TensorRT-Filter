// NOTE: This file was converted from C to C++ to support CUDA/TensorRT integration.
// Original content was in plugin-main.c.

#include <obs-module.h>
#include <graphics/graphics.h>
#include <util/platform.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <windows.h>
#include "trt_runner.h"

#define PLUGIN_VERSION "1.0.0"

// Engine file path (can be made configurable)
#define DEFAULT_ENGINE_PATH "realesrgan_256_fp16.engine"

// Set to 1 to enable verbose per-frame debug logging (normally 0)
#define TRT_VERBOSE 0
#define BLOG_VERBOSE(fmt, ...) do { if (TRT_VERBOSE) blog(LOG_INFO, fmt, ##__VA_ARGS__); } while(0)

/* Per-instance data */
struct trt_filter_data {
    obs_source_t *context;
    
    // Graphics resources
    gs_texrender_t *render_unorm;
    gs_texture_t *output_texture;
    ID3D11Texture2D *output_d3d11_texture; //D3D11 texture with UAV support
    
    // Dimensions
    uint32_t width;
    uint32_t height;
    bool resources_allocated;
    bool target_valid;
    
    // D3D11 device and context
    ID3D11Device *d3d11_device;
    ID3D11DeviceContext *d3d11_context;
    
    // D3D11 compute shaders
    ID3D11ComputeShader *preprocess_shader;
    ID3D11ComputeShader *postprocess_shader;
    
    // D3D11 buffers for TRT input/output
    // Input: FP16 NCHW 1x3x256x256
    // Output: FP16 NCHW 1x3x1024x1024
    ID3D11Buffer *trt_input_buffer;
    ID3D11Buffer *trt_output_buffer;
    ID3D11UnorderedAccessView *trt_input_uav;
    ID3D11UnorderedAccessView *trt_output_uav;
    ID3D11ShaderResourceView *trt_output_srv;
    
    // Constant buffer for shaders
    ID3D11Buffer *constant_buffer;
    ID3D11SamplerState *linear_sampler;
    
    // CUDA resources (runtime API)
    cudaStream_t cuda_stream;
    cudaGraphicsResource_t cuda_trt_input_resource;
    cudaGraphicsResource_t cuda_trt_output_resource;
    void *cuda_trt_input_ptr;
    void *cuda_trt_output_ptr;
    
    // TensorRT runner
    struct trt_runner trt_runner;
    bool trt_initialized;

    // Inference statistics
    uint64_t inference_frame_count;
    uint64_t inference_success_count;
    uint64_t inference_fail_count;

    // Debug metrics settings
    bool metrics_enabled;
    uint32_t metrics_interval;  // Log every N frames
    uint32_t metrics_sample_size;  // 64 or 128
    uint64_t metrics_frame_counter;

    // Metrics GPU resources
    ID3D11ComputeShader *metrics_shader;
    ID3D11Texture2D *metrics_input_small_tex;  // Downsampled input
    ID3D11Texture2D *metrics_output_small_tex;  // Downsampled output
    ID3D11Texture2D *metrics_result_tex;  // 1x1 float texture for MAE/MSE
    ID3D11UnorderedAccessView *metrics_input_small_uav;
    ID3D11UnorderedAccessView *metrics_output_small_uav;
    ID3D11UnorderedAccessView *metrics_result_uav;
    ID3D11ShaderResourceView *metrics_input_small_srv;
    ID3D11ShaderResourceView *metrics_output_small_srv;
    ID3D11Buffer *metrics_staging_buffer;  // For CPU readback
    ID3D11Buffer *metrics_constant_buffer;

    // D3D11 fence for synchronization (currently unused, placeholder for future sync)
    ID3D11Query *d3d11_fence;
    // ID3D11Fence *d3d11_fence_obj;
    HANDLE shared_handle;
    UINT64 fence_value;

    // Debug canary test flags
    bool debug_force_solid_color;
    bool debug_force_opaque;
    bool debug_checkerboard_overlay;
};

/* Display name */
static const char *trt_filter_name(void *unused)
{
    UNUSED_PARAMETER(unused);
    return obs_module_text("TensorRT Inference Filter");
}

/* Helper: Check CUDA runtime errors */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            blog(LOG_ERROR, "[TRT Filter] CUDA error at %s:%d - %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return false; \
        } \
    } while(0)
/* Helper: Check CUDA runtime errors (void function version - doesn't return) */
#define CUDA_CHECK_VOID(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            blog(LOG_ERROR, "[TRT Filter] CUDA error at %s:%d - %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return; \
        } \
    } while(0)
/* Helper: Check CUDA runtime errors */
#define CUDA_RT_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            blog(LOG_ERROR, "[TRT Filter] CUDA runtime error at %s:%d - %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return false; \
        } \
    } while(0)

/* Helper: Check D3D11 errors */
#define D3D11_CHECK(hr, msg) \
    do { \
        if (FAILED(hr)) { \
            blog(LOG_ERROR, "[TRT Filter] D3D11 error: %s (HR=0x%08X)", msg, hr); \
            return false; \
        } \
    } while(0)

/* Compile HLSL compute shader */
static ID3D11ComputeShader *compile_compute_shader(const char *shader_path, const char *entry_point)
{
    ID3D11ComputeShader *shader = nullptr;
    ID3DBlob *blob = nullptr;
    ID3DBlob *error_blob = nullptr;
    
    char *full_path = obs_module_file(shader_path);
    if (!full_path) {
        blog(LOG_ERROR, "[TRT Filter] Failed to get shader path: %s", shader_path);
        return nullptr;
    }
    
    // Convert to wide string for D3DCompileFromFile
    int wlen = MultiByteToWideChar(CP_UTF8, 0, full_path, -1, nullptr, 0);
    wchar_t *wpath = (wchar_t*)bmalloc(wlen * sizeof(wchar_t));
    MultiByteToWideChar(CP_UTF8, 0, full_path, -1, wpath, wlen);
    
    HRESULT hr = D3DCompileFromFile(
        wpath,
        nullptr,
        D3D_COMPILE_STANDARD_FILE_INCLUDE,
        entry_point,
        "cs_5_0",
        D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_WARNINGS_ARE_ERRORS,
        0,
        &blob,
        &error_blob
    );
    
    bfree(wpath);
    bfree(full_path);
    
    if (FAILED(hr)) {
        if (error_blob) {
            blog(LOG_ERROR, "[TRT Filter] Shader compilation error: %s", (char*)error_blob->GetBufferPointer());
            error_blob->Release();
        } else {
            blog(LOG_ERROR, "[TRT Filter] Failed to compile shader: %s (HR=0x%08X)", shader_path, hr);
        }
        return nullptr;
    }
    
    ID3D11Device *device = (ID3D11Device*)gs_get_device_obj();
    if (!device) {
        blog(LOG_ERROR, "[TRT Filter] Failed to get D3D11 device");
        blob->Release();
        return nullptr;
    }
    
    hr = device->CreateComputeShader(
        blob->GetBufferPointer(),
        blob->GetBufferSize(),
        nullptr,
        &shader
    );
    
    blob->Release();
    
    if (FAILED(hr)) {
        blog(LOG_ERROR, "[TRT Filter] Failed to create compute shader");
        return nullptr;
    }
    
    return shader;
}

/* Create/recreate graphics resources */
static bool create_graphics_resources(struct trt_filter_data *filter)
{
    if (filter->render_unorm) {
        gs_texrender_destroy(filter->render_unorm);
        filter->render_unorm = nullptr;
    }
    
    if (filter->output_texture) {
        gs_texture_destroy(filter->output_texture);
        filter->output_texture = nullptr;
    }
    
    if (filter->output_d3d11_texture) {
        filter->output_d3d11_texture->Release();
        filter->output_d3d11_texture = nullptr;
    }
    
    // Create texrender for input (BGRA_UNORM, SRGB)
    filter->render_unorm = gs_texrender_create(GS_BGRA_UNORM, GS_ZS_NONE);
    if (!filter->render_unorm) {
        blog(LOG_ERROR, "[TRT Filter] Failed to create render_unorm texrender");
        return false;
    }
    
    // Get D3D11 device
    ID3D11Device *device = (ID3D11Device *)gs_get_device_obj();
    if (!device) {
        blog(LOG_ERROR, "[TRT Filter] Failed to get D3D11 device");
        gs_texrender_destroy(filter->render_unorm);
        filter->render_unorm = nullptr;
        return false;
    }
    
    // Create output texture using D3D11 directly (with UAV binding support)
    D3D11_TEXTURE2D_DESC tex_desc = {};
    tex_desc.Width = filter->width;
    tex_desc.Height = filter->height;
    tex_desc.MipLevels = 1;
    tex_desc.ArraySize = 1;
    tex_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    tex_desc.SampleDesc.Count = 1;
    tex_desc.SampleDesc.Quality = 0;
    tex_desc.Usage = D3D11_USAGE_DEFAULT;
    tex_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    tex_desc.CPUAccessFlags = 0;
    tex_desc.MiscFlags = 0;
    
    HRESULT hr = device->CreateTexture2D(&tex_desc, nullptr, &filter->output_d3d11_texture);
    if (FAILED(hr)) {
        blog(LOG_ERROR, "[TRT Filter] Failed to create D3D11 output texture (HR=0x%08X)", hr);
        gs_texrender_destroy(filter->render_unorm);
        filter->render_unorm = nullptr;
        return false;
    }
    
    // Create gs_texture for rendering (we'll copy from D3D11 texture to this)
    filter->output_texture = gs_texture_create(
        filter->width, 
        filter->height, 
        GS_BGRA_UNORM, 
        1, 
        nullptr, 
        0);
    if (!filter->output_texture) {
        blog(LOG_ERROR, "[TRT Filter] Failed to create output texture");
        filter->output_d3d11_texture->Release();
        filter->output_d3d11_texture = nullptr;
        gs_texrender_destroy(filter->render_unorm);
        filter->render_unorm = nullptr;
        return false;
    }
    
    return true;
}

/* Create D3D11 buffers for TRT input/output */
static bool create_trt_buffers(struct trt_filter_data *filter)
{
    ID3D11Device *device = filter->d3d11_device;
    
    // TRT input: FP16 NCHW 1x3x256x256
    const uint32_t input_tensor_size = 1 * 3 * 256 * 256;
    // TRT output: FP16 NCHW 1x3x1024x1024
    const uint32_t output_tensor_size = 1 * 3 * 1024 * 1024;
    
    blog(LOG_INFO, "[TRT Filter] Creating TRT buffers - Input: %u elements, Output: %u elements", 
         input_tensor_size, output_tensor_size);
    
    // Create structured buffer for input (RWStructuredBuffer<uint> in shader, stores FP16 in low 16 bits)
    D3D11_BUFFER_DESC input_buffer_desc = {};
    input_buffer_desc.Usage = D3D11_USAGE_DEFAULT;
    input_buffer_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    input_buffer_desc.CPUAccessFlags = 0;
    input_buffer_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    input_buffer_desc.StructureByteStride = sizeof(uint32_t); // we store FP16 in lower 16 bits of uint
    input_buffer_desc.ByteWidth = input_tensor_size * sizeof(uint32_t);
    
    HRESULT hr = device->CreateBuffer(&input_buffer_desc, nullptr, &filter->trt_input_buffer);
    D3D11_CHECK(hr, "Failed to create TRT input buffer");
    
    // Create structured buffer for output (larger size)
    D3D11_BUFFER_DESC output_buffer_desc = {};
    output_buffer_desc.Usage = D3D11_USAGE_DEFAULT;
    output_buffer_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    output_buffer_desc.CPUAccessFlags = 0;
    output_buffer_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    output_buffer_desc.StructureByteStride = sizeof(uint32_t);
    output_buffer_desc.ByteWidth = output_tensor_size * sizeof(uint32_t);
    
    hr = device->CreateBuffer(&output_buffer_desc, nullptr, &filter->trt_output_buffer);
    D3D11_CHECK(hr, "Failed to create TRT output buffer");
    
    // Create UAVs for input
    D3D11_UNORDERED_ACCESS_VIEW_DESC input_uav_desc = {};
    input_uav_desc.Format = DXGI_FORMAT_UNKNOWN;
    input_uav_desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    input_uav_desc.Buffer.FirstElement = 0;
    input_uav_desc.Buffer.NumElements = input_tensor_size;
    input_uav_desc.Buffer.Flags = 0;
    
    hr = device->CreateUnorderedAccessView(filter->trt_input_buffer, &input_uav_desc, &filter->trt_input_uav);
    D3D11_CHECK(hr, "Failed to create TRT input UAV");
    
    // Create UAVs for output
    D3D11_UNORDERED_ACCESS_VIEW_DESC output_uav_desc = {};
    output_uav_desc.Format = DXGI_FORMAT_UNKNOWN;
    output_uav_desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    output_uav_desc.Buffer.FirstElement = 0;
    output_uav_desc.Buffer.NumElements = output_tensor_size;
    output_uav_desc.Buffer.Flags = 0;
    
    hr = device->CreateUnorderedAccessView(filter->trt_output_buffer, &output_uav_desc, &filter->trt_output_uav);
    D3D11_CHECK(hr, "Failed to create TRT output UAV");
    
    // Create SRV for output buffer (for postprocess shader)
    D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = DXGI_FORMAT_UNKNOWN;
    srv_desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
    srv_desc.Buffer.FirstElement = 0;
    srv_desc.Buffer.NumElements = output_tensor_size;
    
    hr = device->CreateShaderResourceView(filter->trt_output_buffer, &srv_desc, &filter->trt_output_srv);
    D3D11_CHECK(hr, "Failed to create TRT output SRV");
    
    // Create constant buffer
    D3D11_BUFFER_DESC cb_desc = {};
    cb_desc.Usage = D3D11_USAGE_DYNAMIC;
    cb_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    cb_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    cb_desc.ByteWidth = 256;  // Aligned to 16 bytes
    
    hr = device->CreateBuffer(&cb_desc, nullptr, &filter->constant_buffer);
    D3D11_CHECK(hr, "Failed to create constant buffer");

    // Create linear clamp sampler for preprocess shader
    D3D11_SAMPLER_DESC sampler_desc = {};
    sampler_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sampler_desc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler_desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler_desc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler_desc.MaxLOD = D3D11_FLOAT32_MAX;
    hr = device->CreateSamplerState(&sampler_desc, &filter->linear_sampler);
    D3D11_CHECK(hr, "Failed to create linear sampler");

    blog(LOG_INFO, "[TRT Filter] Linear sampler created");
    return true;
}

/* Initialize CUDA and register D3D11 resources */
static bool init_cuda_resources(struct trt_filter_data *filter)
{
    // Check device type
    if (gs_get_device_type() != GS_DEVICE_DIRECT3D_11) {
        blog(LOG_ERROR, "[TRT Filter] This filter requires D3D11 backend");
        return false;
    }
    
    // Get D3D11 device
    filter->d3d11_device = (ID3D11Device *)gs_get_device_obj();
    if (!filter->d3d11_device) {
        blog(LOG_ERROR, "[TRT Filter] Failed to get D3D11 device");
        return false;
    }
    
    filter->d3d11_device->GetImmediateContext(&filter->d3d11_context);
    if (!filter->d3d11_context) {
        blog(LOG_ERROR, "[TRT Filter] Failed to get D3D11 context");
        return false;
    }
    
    // Set D3D11 device for CUDA runtime (creates/manages context automatically)
    // Note: This can only be called once per process. If called again, it returns
    // cudaErrorSetOnActiveProcess, which we can safely ignore.
    cudaError_t err = cudaD3D11SetDirect3DDevice(filter->d3d11_device);
    if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess) {
        blog(LOG_ERROR, "[TRT Filter] CUDA error at %s:%d - %s", __FILE__, __LINE__, cudaGetErrorString(err));
        return false;
    }
    if (err == cudaErrorSetOnActiveProcess) {
        blog(LOG_DEBUG, "[TRT Filter] CUDA device already set (multiple filter instances), continuing...");
    }
    
    // Create CUDA stream
    CUDA_CHECK(cudaStreamCreate(&filter->cuda_stream));
    
    // Register TRT buffers with CUDA
    CUDA_CHECK(cudaGraphicsD3D11RegisterResource(
        &filter->cuda_trt_input_resource,
        filter->trt_input_buffer,
        cudaGraphicsRegisterFlagsNone));
    
    CUDA_CHECK(cudaGraphicsD3D11RegisterResource(
        &filter->cuda_trt_output_resource,
        filter->trt_output_buffer,
        cudaGraphicsRegisterFlagsNone));
    
    blog(LOG_INFO, "[TRT Filter] CUDA resources initialized");
    
    return true;
}

/* Initialize TensorRT */
static bool init_tensorrt(struct trt_filter_data *filter)
{
    char *engine_path = obs_module_file(DEFAULT_ENGINE_PATH);
    if (!engine_path) {
        blog(LOG_ERROR, "[TRT Filter] Failed to find engine file: %s", DEFAULT_ENGINE_PATH);
        return false;
    }
    
    blog(LOG_INFO, "[TRT Filter] Loading TensorRT engine from: %s", engine_path);
    
    // Use runtime stream directly (no conversion needed)
    bool success = trt_runner_init(&filter->trt_runner, engine_path, filter->cuda_stream);
    bfree(engine_path);
    
    if (!success) {
        blog(LOG_ERROR, "[TRT Filter] Failed to initialize TensorRT - falling back to passthrough");
        return false;
    }
    
    // Verify tensor names match expected values
    if (strcmp(filter->trt_runner.input_name, "input") != 0) {
        blog(LOG_WARNING, "[TRT Filter] Input tensor name '%s' does not match expected 'input'", 
             filter->trt_runner.input_name);
    }
    if (strcmp(filter->trt_runner.output_name, "output") != 0) {
        blog(LOG_WARNING, "[TRT Filter] Output tensor name '%s' does not match expected 'output'", 
             filter->trt_runner.output_name);
    }
    
    filter->trt_initialized = true;
    
    // Log detailed tensor info
    const nvinfer1::Dims& in_dims = filter->trt_runner.input_dims;
    const nvinfer1::Dims& out_dims = filter->trt_runner.output_dims;
    
    blog(LOG_INFO, "[TRT Filter] TensorRT initialized successfully");
    blog(LOG_INFO, "[TRT Filter] Input tensor '%s': shape [", filter->trt_runner.input_name);
    for (int i = 0; i < in_dims.nbDims; i++) {
        blog(LOG_INFO, "[TRT Filter]   dim[%d] = %d", i, in_dims.d[i]);
    }
    blog(LOG_INFO, "[TRT Filter] ] (type: %s)", 
         filter->trt_runner.input_type == nvinfer1::DataType::kFLOAT ? "FLOAT" : "HALF");
    
    blog(LOG_INFO, "[TRT Filter] Output tensor '%s': shape [", filter->trt_runner.output_name);
    for (int i = 0; i < out_dims.nbDims; i++) {
        blog(LOG_INFO, "[TRT Filter]   dim[%d] = %d", i, out_dims.d[i]);
    }
    blog(LOG_INFO, "[TRT Filter] ] (type: %s)", 
         filter->trt_runner.output_type == nvinfer1::DataType::kFLOAT ? "FLOAT" : "HALF");
    
    // Verify expected shapes
    bool shape_valid = true;
    if (in_dims.nbDims == 4 && 
        in_dims.d[0] == 1 && in_dims.d[1] == 3 && in_dims.d[2] == 256 && in_dims.d[3] == 256) {
        blog(LOG_INFO, "[TRT Filter] Input shape matches expected [1, 3, 256, 256]");
    } else {
        blog(LOG_WARNING, "[TRT Filter] Input shape does not match expected [1, 3, 256, 256]");
        shape_valid = false;
    }
    
    if (out_dims.nbDims == 4 && 
        out_dims.d[0] == 1 && out_dims.d[1] == 3 && out_dims.d[2] == 1024 && out_dims.d[3] == 1024) {
        blog(LOG_INFO, "[TRT Filter] Output shape matches expected [1, 3, 1024, 1024]");
    } else {
        blog(LOG_WARNING, "[TRT Filter] Output shape does not match expected [1, 3, 1024, 1024]");
        shape_valid = false;
    }
    
    if (!shape_valid) {
        blog(LOG_WARNING, "[TRT Filter] Tensor shapes do not match expected values - filter may not work correctly");
    }
    
    blog(LOG_INFO, "[TRT Filter] Ready for inference");
    
    return true;
}

/* Load compute shaders */
static bool load_compute_shaders(struct trt_filter_data *filter)
{
    filter->preprocess_shader = compile_compute_shader("preprocess.hlsl", "CSMain");
    if (!filter->preprocess_shader) {
        blog(LOG_ERROR, "[TRT Filter] Failed to compile preprocess shader");
        return false;
    }
    
    filter->postprocess_shader = compile_compute_shader("postprocess.hlsl", "CSMain");
    if (!filter->postprocess_shader) {
        blog(LOG_ERROR, "[TRT Filter] Failed to compile postprocess shader");
        filter->preprocess_shader->Release();
        filter->preprocess_shader = nullptr;
        return false;
    }
    
    blog(LOG_INFO, "[TRT Filter] Compute shaders loaded");
    return true;
}

/* Cleanup CUDA resources */
static void cleanup_cuda_resources(struct trt_filter_data *filter)
{
    if (filter->cuda_trt_input_resource) {
        cudaGraphicsUnregisterResource(filter->cuda_trt_input_resource);
        filter->cuda_trt_input_resource = nullptr;
    }
    
    if (filter->cuda_trt_output_resource) {
        cudaGraphicsUnregisterResource(filter->cuda_trt_output_resource);
        filter->cuda_trt_output_resource = nullptr;
    }
    
    if (filter->cuda_stream) {
        cudaStreamDestroy(filter->cuda_stream);
        filter->cuda_stream = nullptr;
    }
}

/* Create metrics resources */
static bool create_metrics_resources(struct trt_filter_data *filter)
{
    if (!filter->metrics_enabled) {
        return true;  // Skip if metrics disabled
    }
    
    ID3D11Device *device = filter->d3d11_device;
    uint32_t sample_size = filter->metrics_sample_size;
    HRESULT hr;
    
    // Note: We sample directly from original textures, so no need for downsampled textures
    
    // GPU-side group sums buffer: shader writes per-group float4 here.
    // Max 64 groups (8×8 dispatch for 128×128 sample size) × 16 bytes = 1024 bytes.
    D3D11_BUFFER_DESC buffer_desc = {};
    buffer_desc.Usage = D3D11_USAGE_DEFAULT;
    buffer_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    buffer_desc.CPUAccessFlags = 0;
    buffer_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    buffer_desc.StructureByteStride = sizeof(float) * 4;  // float4 per group
    buffer_desc.ByteWidth = 64 * buffer_desc.StructureByteStride;  // 1024 bytes

    hr = device->CreateBuffer(&buffer_desc, nullptr, &filter->metrics_staging_buffer);
    D3D11_CHECK(hr, "Failed to create metrics GPU group sums buffer");
    
    // Note: We'll create a separate staging buffer for readback when needed
    // For now, we'll use CopyResource to a staging buffer created on-demand
    
    // Create constant buffer for metrics shader
    D3D11_BUFFER_DESC cb_desc = {};
    cb_desc.Usage = D3D11_USAGE_DYNAMIC;
    cb_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    cb_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    cb_desc.ByteWidth = 64;  // Aligned to 16 bytes
    
    hr = device->CreateBuffer(&cb_desc, nullptr, &filter->metrics_constant_buffer);
    D3D11_CHECK(hr, "Failed to create metrics constant buffer");
    
    // Compile metrics shader
    filter->metrics_shader = compile_compute_shader("metrics.hlsl", "CSMain");
    if (!filter->metrics_shader) {
        blog(LOG_ERROR, "[TRT Filter] Failed to compile metrics shader");
        return false;
    }
    
    blog(LOG_INFO, "[TRT Filter] Metrics resources created (sample_size=%u)", sample_size);
    return true;
}

/* Cleanup metrics resources */
static void cleanup_metrics_resources(struct trt_filter_data *filter)
{
    if (filter->metrics_shader) {
        filter->metrics_shader->Release();
        filter->metrics_shader = nullptr;
    }
    
    if (filter->metrics_staging_buffer) {
        filter->metrics_staging_buffer->Release();
        filter->metrics_staging_buffer = nullptr;
    }
    
    if (filter->metrics_constant_buffer) {
        filter->metrics_constant_buffer->Release();
        filter->metrics_constant_buffer = nullptr;
    }
}

/* Cleanup D3D11 resources */
static void cleanup_d3d11_resources(struct trt_filter_data *filter)
{
    // Cleanup metrics resources
    cleanup_metrics_resources(filter);
    
    if (filter->trt_output_srv) {
        filter->trt_output_srv->Release();
        filter->trt_output_srv = nullptr;
    }
    
    if (filter->trt_input_uav) {
        filter->trt_input_uav->Release();
        filter->trt_input_uav = nullptr;
    }
    
    if (filter->trt_output_uav) {
        filter->trt_output_uav->Release();
        filter->trt_output_uav = nullptr;
    }
    
    if (filter->trt_input_buffer) {
        filter->trt_input_buffer->Release();
        filter->trt_input_buffer = nullptr;
    }
    
    if (filter->trt_output_buffer) {
        filter->trt_output_buffer->Release();
        filter->trt_output_buffer = nullptr;
    }
    
    if (filter->constant_buffer) {
        filter->constant_buffer->Release();
        filter->constant_buffer = nullptr;
    }

    if (filter->linear_sampler) {
        filter->linear_sampler->Release();
        filter->linear_sampler = nullptr;
    }
    
    if (filter->preprocess_shader) {
        filter->preprocess_shader->Release();
        filter->preprocess_shader = nullptr;
    }
    
    if (filter->postprocess_shader) {
        filter->postprocess_shader->Release();
        filter->postprocess_shader = nullptr;
    }
    
    if (filter->d3d11_fence) {
        filter->d3d11_fence->Release();
        filter->d3d11_fence = nullptr;
    }
    
    // if (filter->d3d11_fence_obj) {
    //     filter->d3d11_fence_obj->Release();
    //     filter->d3d11_fence_obj = nullptr;
    // }
    
    if (filter->shared_handle) {
        CloseHandle(filter->shared_handle);
        filter->shared_handle = nullptr;
    }
    
    if (filter->d3d11_context) {
        filter->d3d11_context->Release();
        filter->d3d11_context = nullptr;
    }
}

/* Destructor */
static void trt_filter_destroy(void *data)
{
    auto *filter = static_cast<trt_filter_data *>(data);
    
    obs_enter_graphics();
    
    // Cleanup TensorRT
    if (filter->trt_initialized) {
        trt_runner_destroy(&filter->trt_runner);
    }
    
    // Cleanup CUDA resources
    cleanup_cuda_resources(filter);
    
    // Cleanup D3D11 resources
    cleanup_d3d11_resources(filter);
    
    // Cleanup graphics resources
    if (filter->render_unorm) {
        gs_texrender_destroy(filter->render_unorm);
    }
    if (filter->output_texture) {
        gs_texture_destroy(filter->output_texture);
    }
    if (filter->output_d3d11_texture) {
        filter->output_d3d11_texture->Release();
    }
    
    obs_leave_graphics();
    
    bfree(filter);
}

/* Debug helper: Log effect information */
static void debug_log_effect_info(gs_effect_t *effect, const char *effect_name)
{
    if (!effect) {
        blog(LOG_WARNING, "[TRT Filter] DEBUG - Effect '%s': effect pointer is NULL", effect_name);
        return;
    }
    
    blog(LOG_INFO, "[TRT Filter] DEBUG - Effect '%s' info:", effect_name);
    blog(LOG_INFO, "  - Effect pointer: %p", effect);
    
    // Try common technique names (OBS doesn't expose enumeration)
    const char *common_techniques[] = {"Draw", "Solid", "Default", nullptr};
    blog(LOG_INFO, "  - Testing common technique names:");
    for (int i = 0; common_techniques[i]; i++) {
        gs_technique_t *tech = gs_effect_get_technique(effect, common_techniques[i]);
        blog(LOG_INFO, "  - Technique '%s': %s", common_techniques[i], tech ? "EXISTS" : "NOT FOUND");
    }
    
    // Check for common parameters
    const char *param_names[] = {"image", "color", "alpha", "uv_size", "uv_offset", nullptr};
    for (int i = 0; param_names[i]; i++) {
        gs_eparam_t *param = gs_effect_get_param_by_name(effect, param_names[i]);
        blog(LOG_INFO, "  - Parameter '%s': %s", param_names[i], param ? "EXISTS" : "NOT FOUND");
    }
}

/* Helper: Get first valid technique name from effect by trying common names */
static const char *get_first_technique_name(gs_effect_t *effect, bool is_solid)
{
    if (!effect) return nullptr;
    
    // Try technique names in order of likelihood
    const char *techniques_to_try[] = {
        is_solid ? "Solid" : "Draw",  // Most likely for each effect type
        "Draw",                        // Fallback
        "Default",                     // Alternative
        nullptr
    };
    
    for (int i = 0; techniques_to_try[i]; i++) {
        gs_technique_t *tech = gs_effect_get_technique(effect, techniques_to_try[i]);
        if (tech) {
            return techniques_to_try[i];
        }
    }
    
    blog(LOG_WARNING, "[TRT Filter] DEBUG - No valid technique found for effect");
    return nullptr;
}

/* Get filter properties */
static obs_properties_t *trt_filter_properties(void *unused)
{
    UNUSED_PARAMETER(unused);
    
    obs_properties_t *props = obs_properties_create();
    
    obs_properties_add_bool(props, "metrics_enabled", "Enable Debug Difference Metrics");
    obs_properties_add_int(props, "metrics_interval", "Metric Interval (frames)", 1, 1000, 1);
    obs_properties_add_int(props, "metrics_sample_size", "Metric Sample Size", 64, 128, 64);
    
    obs_properties_add_bool(props, "debug_force_solid_color", "DEBUG: Force Solid Magenta (bypass texture)");
    obs_properties_add_bool(props, "debug_force_opaque", "DEBUG: Force Opaque Alpha");
    obs_properties_add_bool(props, "debug_checkerboard_overlay", "DEBUG: Checkerboard Overlay");
    
    return props;
}

/* Get default settings */
static void trt_filter_defaults(obs_data_t *settings)
{
    obs_data_set_default_bool(settings, "metrics_enabled", false);
    obs_data_set_default_int(settings, "metrics_interval", 60);
    obs_data_set_default_int(settings, "metrics_sample_size", 64);
    
    obs_data_set_default_bool(settings, "debug_force_solid_color", false);
    obs_data_set_default_bool(settings, "debug_force_opaque", false);
    obs_data_set_default_bool(settings, "debug_checkerboard_overlay", false);
}

/* Update settings */
static void trt_filter_update(void *data, obs_data_t *settings)
{
    auto *filter = static_cast<trt_filter_data *>(data);
    
    filter->metrics_enabled = obs_data_get_bool(settings, "metrics_enabled");
    filter->metrics_interval = (uint32_t)obs_data_get_int(settings, "metrics_interval");
    uint32_t sample_size = (uint32_t)obs_data_get_int(settings, "metrics_sample_size");
    // Clamp to 64 or 128
    filter->metrics_sample_size = (sample_size == 128) ? 128 : 64;
    
    filter->debug_force_solid_color = obs_data_get_bool(settings, "debug_force_solid_color");
    filter->debug_force_opaque = obs_data_get_bool(settings, "debug_force_opaque");
    filter->debug_checkerboard_overlay = obs_data_get_bool(settings, "debug_checkerboard_overlay");
    
    blog(LOG_INFO, "[TRT Filter] Metrics settings updated: enabled=%d, interval=%u, sample_size=%u",
         filter->metrics_enabled, filter->metrics_interval, filter->metrics_sample_size);
    
    // Create metrics resources if enabled and not already created
    if (filter->metrics_enabled && !filter->metrics_shader && filter->resources_allocated) {
        obs_enter_graphics();
        if (!create_metrics_resources(filter)) {
            blog(LOG_WARNING, "[TRT Filter] Failed to create metrics resources, continuing without metrics");
            filter->metrics_enabled = false;
        }
        obs_leave_graphics();
    }
    
    // Reset counter when settings change
    filter->metrics_frame_counter = 0;
}

/* Constructor */
static void *trt_filter_create(obs_data_t *settings, obs_source_t *source)
{
    auto *filter = static_cast<trt_filter_data *>(bzalloc(sizeof(trt_filter_data)));
    filter->context = source;
    filter->width = 0;
    filter->height = 0;
    filter->resources_allocated = false;
    filter->target_valid = false;
    filter->trt_initialized = false;

    // Initialize inference statistics
    filter->inference_frame_count = 0;
    filter->inference_success_count = 0;
    filter->inference_fail_count = 0;
    
    // Initialize metrics settings
    filter->metrics_enabled = false;
    filter->metrics_interval = 60;
    filter->metrics_sample_size = 64;
    filter->metrics_frame_counter = 0;
    
    // Initialize debug canary flags
    filter->debug_force_solid_color = false;
    filter->debug_force_opaque = false;
    filter->debug_checkerboard_overlay = false;
    
    // Initialize metrics resources to nullptr
    filter->metrics_shader = nullptr;
    filter->metrics_input_small_tex = nullptr;
    filter->metrics_output_small_tex = nullptr;
    filter->metrics_result_tex = nullptr;
    filter->metrics_input_small_uav = nullptr;
    filter->metrics_output_small_uav = nullptr;
    filter->metrics_result_uav = nullptr;
    filter->metrics_input_small_srv = nullptr;
    filter->metrics_output_small_srv = nullptr;
    filter->metrics_staging_buffer = nullptr;
    filter->metrics_constant_buffer = nullptr;
    
    // Load settings if provided
    if (settings) {
        trt_filter_update(filter, settings);
    }
    
    // CUDA runtime doesn't need explicit initialization
    // It will be initialized when we call cudaD3D11SetDirect3DDevice
    
    return filter;
}

/* Tick function - track size changes */
static void trt_filter_tick(void *data, float seconds)
{
    UNUSED_PARAMETER(seconds);
    
    auto *filter = static_cast<trt_filter_data *>(data);
    
    if (!obs_filter_get_target(filter->context)) {
        filter->target_valid = false;
        return;
    }
    
    obs_source_t *target = obs_filter_get_target(filter->context);
    const uint32_t cx = obs_source_get_base_width(target);
    const uint32_t cy = obs_source_get_base_height(target);
    
    filter->target_valid = true;
    
    if (!cx || !cy) {
        filter->target_valid = false;
        return;
    }
    
    // Check if size changed
    if (cx != filter->width || cy != filter->height) {
        filter->resources_allocated = false;
        filter->width = cx;
        filter->height = cy;
    }
    
    // Recreate resources if needed
    if (!filter->resources_allocated) {
        obs_enter_graphics();
        
        // Create graphics resources
        if (!create_graphics_resources(filter)) {
            filter->resources_allocated = false;
            obs_leave_graphics();
            return;
        }
        
        // Get D3D11 device
        filter->d3d11_device = (ID3D11Device *)gs_get_device_obj();
        if (!filter->d3d11_device) {
            blog(LOG_ERROR, "[TRT Filter] Failed to get D3D11 device");
            filter->resources_allocated = false;
            obs_leave_graphics();
            return;
        }
        
        filter->d3d11_device->GetImmediateContext(&filter->d3d11_context);
        
        // Create TRT buffers (only once, sizes are fixed: input 256×256, output 1024×1024)
        if (!filter->trt_input_buffer) {
            if (!create_trt_buffers(filter)) {
                filter->resources_allocated = false;
                obs_leave_graphics();
                return;
            }
        }
        
        // Load compute shaders (only once)
        if (!filter->preprocess_shader) {
            if (!load_compute_shaders(filter)) {
                filter->resources_allocated = false;
                obs_leave_graphics();
                return;
            }
        }
        
        // Create metrics resources if enabled (only once)
        if (filter->metrics_enabled && !filter->metrics_shader) {
            if (!create_metrics_resources(filter)) {
                blog(LOG_WARNING, "[TRT Filter] Failed to create metrics resources, continuing without metrics");
                filter->metrics_enabled = false;  // Disable metrics if creation fails
            }
        }
        
        // Initialize CUDA resources (only once)
        if (!filter->cuda_stream) {
            if (!init_cuda_resources(filter)) {
                filter->resources_allocated = false;
                obs_leave_graphics();
                return;
            }
        }
        
        // Initialize TensorRT (only once)
        if (!filter->trt_initialized) {
            if (!init_tensorrt(filter)) {
                filter->resources_allocated = false;
                obs_leave_graphics();
                return;
            }
        }
        
        filter->resources_allocated = true;
        
        // Log resize policies
        blog(LOG_INFO, "[TRT Filter] Graphics resources allocated");
        blog(LOG_INFO, "[TRT Filter] Preprocess resize policy: W×H (%ux%u) → 256×256 (centered crop/resize)", 
             filter->width, filter->height);
        blog(LOG_INFO, "[TRT Filter] Postprocess resample policy: 1024×1024 → W×H (%ux%u) (bilinear interpolation)", 
             filter->width, filter->height);
        blog(LOG_INFO, "[TRT Filter] Final output texture size: %ux%u (matches input size)", 
             filter->width, filter->height);
        
        obs_leave_graphics();
    }
}

/* Per-frame render */
static void trt_filter_render(void *data, gs_effect_t *effect)
{
    UNUSED_PARAMETER(effect);
    
    auto *filter = static_cast<trt_filter_data *>(data);
    
    if (!filter->target_valid || !filter->resources_allocated || !filter->trt_initialized) {
        blog(LOG_WARNING, "[TRT Filter] Early return: target_valid=%d, resources_allocated=%d, trt_initialized=%d",
             filter->target_valid, filter->resources_allocated, filter->trt_initialized);
        obs_source_skip_video_filter(filter->context);
        return;
    }

    obs_source_t *target = obs_filter_get_target(filter->context);
    obs_source_t *parent = obs_filter_get_parent(filter->context);

    if (!target || !parent) {
        blog(LOG_WARNING, "[TRT Filter] Early return: target=%p, parent=%p", target, parent);
        obs_source_skip_video_filter(filter->context);
        return;
    }
    
    const uint32_t target_flags = obs_source_get_output_flags(target);
    bool custom_draw = (target_flags & OBS_SOURCE_CUSTOM_DRAW) != 0;
    bool async = (target_flags & OBS_SOURCE_ASYNC) != 0;
    
    // Step 1: Render upstream source to texrender (BGRA_UNORM, SRGB)
    gs_texrender_t *render_unorm = filter->render_unorm;
    gs_texrender_reset(render_unorm);
    gs_blend_state_push();
    gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
    
    if (gs_texrender_begin_with_color_space(render_unorm, filter->width, filter->height, GS_CS_SRGB)) {
        struct vec4 clear_color;
        vec4_zero(&clear_color);
        gs_clear(GS_CLEAR_COLOR, &clear_color, 0.0f, 0);
        
        gs_ortho(0.0f, (float)filter->width, 0.0f, (float)filter->height, -100.0f, 100.0f);
        
        // Render upstream source
        if (target == parent && !custom_draw && !async) {
            obs_source_default_render(target);
        } else {
            obs_source_video_render(target);
        }
        
        gs_texrender_end(render_unorm);
    }
    
    gs_blend_state_pop();
    
    // Step 2: Preprocess with compute shader (BGRA -> FP16 NCHW)
    gs_texture_t *input_texture = gs_texrender_get_texture(render_unorm);
    ID3D11Texture2D *input_d3d11 = (ID3D11Texture2D *)gs_texture_get_obj(input_texture);
    
    // Create SRV for input texture using the texture's actual format
    if (!input_d3d11) {
        blog(LOG_ERROR, "[TRT Filter] Preprocess: input_d3d11 is null (gs_texrender_get_texture returned null)");
        obs_source_skip_video_filter(filter->context);
        return;
    }

    D3D11_TEXTURE2D_DESC input_tex_desc = {};
    input_d3d11->GetDesc(&input_tex_desc);

    static bool srv_format_logged = false;
    if (!srv_format_logged) {
        blog(LOG_INFO, "[TRT Filter] Preprocess input texture: %ux%u format=%d",
             input_tex_desc.Width, input_tex_desc.Height, (int)input_tex_desc.Format);

        // One-time center-pixel readback to verify source texture content
        D3D11_TEXTURE2D_DESC staging_desc = input_tex_desc;
        staging_desc.Usage          = D3D11_USAGE_STAGING;
        staging_desc.BindFlags      = 0;
        staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        staging_desc.MiscFlags      = 0;
        staging_desc.MipLevels      = 1;
        ID3D11Texture2D *staging = nullptr;
        HRESULT st_hr = filter->d3d11_device->CreateTexture2D(&staging_desc, nullptr, &staging);
        if (SUCCEEDED(st_hr) && staging) {
            filter->d3d11_context->CopyResource(staging, input_d3d11);
            D3D11_MAPPED_SUBRESOURCE st_mapped = {};
            if (SUCCEEDED(filter->d3d11_context->Map(staging, 0, D3D11_MAP_READ, 0, &st_mapped))) {
                uint32_t cx = input_tex_desc.Width  / 2;
                uint32_t cy = input_tex_desc.Height / 2;
                const uint8_t *row = (const uint8_t *)st_mapped.pData + cy * st_mapped.RowPitch;
                const uint8_t *px  = row + cx * 4;  // 4 bytes per BGRA pixel
                blog(LOG_INFO, "[TRT Filter] Source texture center pixel (BGRA): B=%d G=%d R=%d A=%d",
                     px[0], px[1], px[2], px[3]);
                filter->d3d11_context->Unmap(staging, 0);
            } else {
                blog(LOG_WARNING, "[TRT Filter] Source texture readback: Map failed");
            }
            staging->Release();
        } else {
            blog(LOG_WARNING, "[TRT Filter] Source texture readback: CreateTexture2D failed (HR=0x%08X)", st_hr);
        }

        srv_format_logged = true;
    }

    ID3D11ShaderResourceView *input_srv = nullptr;
    D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = input_tex_desc.Format;  // match actual texture format, not assumed UNORM
    srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srv_desc.Texture2D.MipLevels = 1;
    srv_desc.Texture2D.MostDetailedMip = 0;
    HRESULT srv_hr = filter->d3d11_device->CreateShaderResourceView(input_d3d11, &srv_desc, &input_srv);
    if (FAILED(srv_hr)) {
        blog(LOG_ERROR, "[TRT Filter] Preprocess: CreateShaderResourceView failed (HR=0x%08X, format=%d)",
             srv_hr, (int)input_tex_desc.Format);
        obs_source_skip_video_filter(filter->context);
        return;
    }
    
    // Update constant buffer
    struct {
        uint32_t input_size[2];
        uint32_t output_size[2];
        float scale[2];
        float offset[2];
    } constants;
    
    constants.input_size[0] = filter->width;
    constants.input_size[1] = filter->height;
    constants.output_size[0] = 256;
    constants.output_size[1] = 256;
    
    // Preprocess resize policy: Centered crop/resize from W×H → 256×256
    // This maintains aspect ratio by scaling to fit the smaller dimension, then cropping the larger dimension
    // Independent per-axis scale: maps the full W×H source to the 256×256 input without cropping.
    // The postprocess does the inverse (1024×1024 → W×H with independent axes), so aspect ratio
    // is preserved end-to-end even though the intermediate 256×256 tensor is squashed.
    constants.scale[0] = (float)filter->width  / 256.0f;
    constants.scale[1] = (float)filter->height / 256.0f;
    constants.offset[0] = 0.0f;
    constants.offset[1] = 0.0f;
    
    D3D11_MAPPED_SUBRESOURCE mapped;
    filter->d3d11_context->Map(filter->constant_buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    memcpy(mapped.pData, &constants, sizeof(constants));
    filter->d3d11_context->Unmap(filter->constant_buffer, 0);
    
    // Unbind any active OM render targets before binding CS SRV.
    // D3D11 hazard detection silently nulls CS t0 if input_d3d11 is still
    // bound as an RTV (OBS leaves it bound after gs_texrender_end restores
    // the previous RT stack). obs_source_process_filter_begin (Step 7) will
    // re-establish the correct RT before drawing.
    filter->d3d11_context->OMSetRenderTargets(0, nullptr, nullptr);

    // Set compute shader state
    filter->d3d11_context->CSSetShader(filter->preprocess_shader, nullptr, 0);
    filter->d3d11_context->CSSetConstantBuffers(0, 1, &filter->constant_buffer);
    filter->d3d11_context->CSSetShaderResources(0, 1, &input_srv);
    filter->d3d11_context->CSSetUnorderedAccessViews(0, 1, &filter->trt_input_uav, nullptr);
    filter->d3d11_context->CSSetSamplers(0, 1, &filter->linear_sampler);

    // One-time binding probe
    static bool binding_logged = false;
    if (!binding_logged) {
        blog(LOG_INFO, "[TRT Filter] Preprocess binding: srv=%p sampler=%p uav=%p shader=%p",
             input_srv, filter->linear_sampler,
             filter->trt_input_uav, filter->preprocess_shader);
        binding_logged = true;
    }

    // Dispatch compute shader (256x256 = 16x16 thread groups)
    filter->d3d11_context->Dispatch(16, 16, 1);

    // Unbind resources
    ID3D11ShaderResourceView *null_srv = nullptr;
    ID3D11UnorderedAccessView *null_uav = nullptr;
    ID3D11SamplerState *null_sampler = nullptr;
    filter->d3d11_context->CSSetShaderResources(0, 1, &null_srv);
    filter->d3d11_context->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
    filter->d3d11_context->CSSetSamplers(0, 1, &null_sampler);
    
    if (input_srv) {
        input_srv->Release();
    }
    
    // Step 3: Synchronize D3D11 -> CUDA
    // Flush D3D11 commands
    filter->d3d11_context->Flush();
    
    // Map CUDA resources
    cudaGraphicsResource_t resources[2] = {
        filter->cuda_trt_input_resource,
        filter->cuda_trt_output_resource
    };
    
    CUDA_CHECK_VOID(cudaGraphicsMapResources(2, resources, filter->cuda_stream));
    
    // Get CUDA device pointers
    void *d_input_ptr = nullptr;
    void *d_output_ptr = nullptr;
    size_t input_size = 0;
    size_t output_size = 0;
    
    CUDA_CHECK_VOID(cudaGraphicsResourceGetMappedPointer(&d_input_ptr, &input_size, filter->cuda_trt_input_resource));
    CUDA_CHECK_VOID(cudaGraphicsResourceGetMappedPointer(&d_output_ptr, &output_size, filter->cuda_trt_output_resource));
    
    filter->cuda_trt_input_ptr = d_input_ptr;
    filter->cuda_trt_output_ptr = d_output_ptr;

    // Buffer probe: log first 8 floats at frame 1 and every 120 frames
    static uint64_t probe_frame = 0;
    probe_frame++;
    bool do_probe = (probe_frame == 1 || probe_frame % 120 == 0);
    if (do_probe) {
        float in_vals[8] = {};
        cudaMemcpy(in_vals, d_input_ptr, 8 * sizeof(float), cudaMemcpyDeviceToHost);
        blog(LOG_INFO, "[TRT Filter] TRT input[0..7]: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f",
             in_vals[0], in_vals[1], in_vals[2], in_vals[3],
             in_vals[4], in_vals[5], in_vals[6], in_vals[7]);
    }

    // Step 4: Run TensorRT inference
    filter->inference_frame_count++;

    trt_runner_set_buffers(&filter->trt_runner, filter->cuda_trt_input_ptr, filter->cuda_trt_output_ptr);
    if (!trt_runner_enqueue(&filter->trt_runner)) {
        filter->inference_fail_count++;
        blog(LOG_ERROR, "[TRT Filter] TensorRT inference failed (frame %llu, total fails: %llu)",
             (unsigned long long)filter->inference_frame_count,
             (unsigned long long)filter->inference_fail_count);
        cudaGraphicsUnmapResources(2, resources, filter->cuda_stream);
        obs_source_skip_video_filter(filter->context);
        return;
    }

    filter->inference_success_count++;
    if (filter->inference_frame_count % 60 == 0) {
        blog(LOG_INFO,
             "[TRT Filter] Inference stats - total: %llu, success: %llu, failed: %llu",
             (unsigned long long)filter->inference_frame_count,
             (unsigned long long)filter->inference_success_count,
             (unsigned long long)filter->inference_fail_count);
    }
    
    // Step 5: Synchronize CUDA -> D3D11
    CUDA_CHECK_VOID(cudaStreamSynchronize(filter->cuda_stream));

    if (do_probe) {
        float out_vals[8] = {};
        cudaMemcpy(out_vals, d_output_ptr, 8 * sizeof(float), cudaMemcpyDeviceToHost);
        blog(LOG_INFO, "[TRT Filter] TRT output[0..7]: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f",
             out_vals[0], out_vals[1], out_vals[2], out_vals[3],
             out_vals[4], out_vals[5], out_vals[6], out_vals[7]);
    }

    // Unmap CUDA resources
    CUDA_CHECK_VOID(cudaGraphicsUnmapResources(2, resources, filter->cuda_stream));
    
    // Step 6: Postprocess with compute shader (FP16 NCHW -> BGRA)
    // Use the D3D11 texture with UAV support
    ID3D11Texture2D *output_d3d11 = filter->output_d3d11_texture;
    
    // Create UAV for output texture
    ID3D11UnorderedAccessView *output_uav = nullptr;
    D3D11_UNORDERED_ACCESS_VIEW_DESC uav_desc = {};
    uav_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    uav_desc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
    uav_desc.Texture2D.MipSlice = 0;
    
    HRESULT hr = filter->d3d11_device->CreateUnorderedAccessView(output_d3d11, &uav_desc, &output_uav);
    if (FAILED(hr)) {
        blog(LOG_ERROR, "[TRT Filter] Failed to create output UAV (HR=0x%08X)", hr);
        obs_source_skip_video_filter(filter->context);
        return;
    }
    
    // Update constant buffer for postprocess
    // TRT output is 1024x1024, we need to resample to W×H
    struct {
        uint32_t output_size[2];      // Final output texture size (W×H)
        uint32_t tensor_size[2];       // TRT output tensor size (1024×1024)
    } post_constants;
    
    post_constants.output_size[0] = filter->width;
    post_constants.output_size[1] = filter->height;
    post_constants.tensor_size[0] = 1024;
    post_constants.tensor_size[1] = 1024;
    
    filter->d3d11_context->Map(filter->constant_buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    memcpy(mapped.pData, &post_constants, sizeof(post_constants));
    filter->d3d11_context->Unmap(filter->constant_buffer, 0);
    
    // Set compute shader state
    filter->d3d11_context->CSSetShader(filter->postprocess_shader, nullptr, 0);
    filter->d3d11_context->CSSetConstantBuffers(0, 1, &filter->constant_buffer);
    filter->d3d11_context->CSSetShaderResources(0, 1, &filter->trt_output_srv);
    filter->d3d11_context->CSSetUnorderedAccessViews(0, 1, &output_uav, nullptr);
    
    // Dispatch compute shader
    uint32_t groups_x = (filter->width + 15) / 16;
    uint32_t groups_y = (filter->height + 15) / 16;

    // Create query BEFORE dispatch for proper synchronization
    ID3D11Query *pQuery = nullptr;
    D3D11_QUERY_DESC queryDesc = {};
    queryDesc.Query = D3D11_QUERY_EVENT;
    HRESULT query_hr = filter->d3d11_device->CreateQuery(&queryDesc, &pQuery);

    filter->d3d11_context->Dispatch(groups_x, groups_y, 1);
    
    // Unbind resources FIRST (important for state transitions)
    filter->d3d11_context->CSSetShaderResources(0, 1, &null_srv);
    filter->d3d11_context->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
    
    if (output_uav) {
        output_uav->Release();
    }
    
    // Barrier: Ensure compute shader completes before copying
    // End query after unbinding UAVs and wait for GPU completion
    if (SUCCEEDED(query_hr) && pQuery) {
        filter->d3d11_context->End(pQuery);
        filter->d3d11_context->Flush();
        
        // Wait with timeout to avoid infinite loop
        BOOL queryData = FALSE;
        int attempts = 0;
        const int max_attempts = 10000;
        HRESULT hr = S_FALSE;
        while ((hr = filter->d3d11_context->GetData(pQuery, &queryData, sizeof(BOOL), 0)) == S_FALSE && attempts < max_attempts) {
            Sleep(0);  // Yield to other threads
            attempts++;
        }
        
        if (hr != S_OK) {
            blog(LOG_WARNING, "[TRT Filter] Postprocess query wait failed or timed out (HR=0x%08X, attempts=%d)", hr, attempts);
        }

        pQuery->Release();
    } else {
        blog(LOG_WARNING, "[TRT Filter] Failed to create postprocess sync query");
    }
    
    // Additional resource state barrier using async query
    ID3D11Query *pAsyncQuery = nullptr;
    D3D11_QUERY_DESC asyncQueryDesc = {};
    asyncQueryDesc.Query = D3D11_QUERY_EVENT;
    HRESULT async_hr = filter->d3d11_device->CreateQuery(&asyncQueryDesc, &pAsyncQuery);
    
    if (SUCCEEDED(async_hr) && pAsyncQuery) {
        filter->d3d11_context->End(pAsyncQuery);
        filter->d3d11_context->Flush();
        
        // Wait for all previous GPU work to complete
        BOOL asyncData = FALSE;
        int async_attempts = 0;
        const int max_async_attempts = 10000;
        HRESULT async_hr_wait = S_FALSE;
        while ((async_hr_wait = filter->d3d11_context->GetData(pAsyncQuery, &asyncData, sizeof(BOOL), 0)) == S_FALSE && async_attempts < max_async_attempts) {
            Sleep(0);  // Busy wait with yield
            async_attempts++;
        }
        
        if (async_hr_wait != S_OK) {
            blog(LOG_WARNING, "[TRT Filter] Async barrier query wait failed (HR=0x%08X, attempts=%d)", async_hr_wait, async_attempts);
        }
        
        pAsyncQuery->Release();
    }
    
    // Final flush to ensure all commands are submitted
    filter->d3d11_context->Flush();

    // Copy D3D11 texture to gs_texture for rendering
    ID3D11Texture2D *gs_tex_d3d11 = (ID3D11Texture2D *)gs_texture_get_obj(filter->output_texture);

    if (gs_tex_d3d11) {
        filter->d3d11_context->CopyResource(gs_tex_d3d11, output_d3d11);

        // Wait for copy to complete before drawing
        ID3D11Query *copyCompleteQuery = nullptr;
        D3D11_QUERY_DESC copyQueryDesc = {};
        copyQueryDesc.Query = D3D11_QUERY_EVENT;
        HRESULT copyQuery_hr = filter->d3d11_device->CreateQuery(&copyQueryDesc, &copyCompleteQuery);
        if (SUCCEEDED(copyQuery_hr) && copyCompleteQuery) {
            filter->d3d11_context->End(copyCompleteQuery);
            filter->d3d11_context->Flush();

            BOOL copyComplete = FALSE;
            int copy_wait_attempts = 0;
            const int max_copy_wait_attempts = 10000;
            HRESULT copy_hr = S_FALSE;
            while ((copy_hr = filter->d3d11_context->GetData(copyCompleteQuery, &copyComplete, sizeof(BOOL), 0)) == S_FALSE && copy_wait_attempts < max_copy_wait_attempts) {
                Sleep(0);
                copy_wait_attempts++;
            }

            if (copy_hr != S_OK) {
                blog(LOG_WARNING, "[TRT Filter] Copy completion query wait failed (HR=0x%08X, attempts=%d)", copy_hr, copy_wait_attempts);
            }

            copyCompleteQuery->Release();
        }

        filter->d3d11_context->Flush();
    } else {
        blog(LOG_ERROR, "[TRT Filter] gs_texture_get_obj returned null - cannot copy output texture");
    }
    
    // Step 7: Draw output texture
    bool begin_result = obs_source_process_filter_begin(filter->context, GS_BGRA_UNORM, OBS_ALLOW_DIRECT_RENDERING);
    if (!begin_result) {
        blog(LOG_WARNING, "[TRT Filter] obs_source_process_filter_begin() returned false");
        return;
    }

    gs_effect_t *default_effect = nullptr;
    gs_eparam_t *image_param = nullptr;
    const char *technique_name = nullptr;

    if (filter->debug_force_solid_color) {
        default_effect = obs_get_base_effect(OBS_EFFECT_SOLID);
        technique_name = get_first_technique_name(default_effect, true);
        if (!technique_name) {
            blog(LOG_ERROR, "[TRT Filter] Failed to get technique name from OBS_EFFECT_SOLID");
            obs_source_process_filter_end(filter->context, default_effect, 0, 0);
            return;
        }
        gs_eparam_t *color_param = gs_effect_get_param_by_name(default_effect, "color");
        if (color_param) {
            struct vec4 magenta = {1.0f, 0.0f, 1.0f, 1.0f};
            gs_effect_set_vec4(color_param, &magenta);
        } else {
            static bool color_param_warned = false;
            if (!color_param_warned) {
                blog(LOG_WARNING, "[TRT Filter] 'color' param not found in OBS_EFFECT_SOLID");
                color_param_warned = true;
            }
        }
        image_param = nullptr;
        filter->debug_force_opaque = true;
    } else {
        default_effect = obs_get_base_effect(OBS_EFFECT_DEFAULT);
        technique_name = get_first_technique_name(default_effect, false);
        if (!technique_name) {
            blog(LOG_ERROR, "[TRT Filter] Failed to get technique name from OBS_EFFECT_DEFAULT");
            obs_source_process_filter_end(filter->context, default_effect, 0, 0);
            return;
        }
        image_param = gs_effect_get_param_by_name(default_effect, "image");
        if (!image_param) {
            static bool image_param_warned = false;
            if (!image_param_warned) {
                blog(LOG_WARNING, "[TRT Filter] 'image' param not found in OBS_EFFECT_DEFAULT");
                image_param_warned = true;
            }
        }
    }

    // Set texture via effect parameter (unless using solid color)
    if (!filter->debug_force_solid_color) {
        if (image_param) {
            gs_effect_set_texture(image_param, filter->output_texture);
        } else {
            static bool image_set_warned = false;
            if (!image_set_warned) {
                blog(LOG_WARNING, "[TRT Filter] image_param is NULL, skipping gs_effect_set_texture()");
                image_set_warned = true;
            }
        }
    }

    // Blend state setup
    gs_blend_state_push();
    if (filter->debug_force_opaque) {
        gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
    } else {
        gs_blend_function(GS_BLEND_ONE, GS_BLEND_INVSRCALPHA);
    }

    if (!technique_name) {
        blog(LOG_ERROR, "[TRT Filter] technique_name is NULL, cannot draw");
        gs_blend_state_pop();
        obs_source_process_filter_end(filter->context, default_effect, 0, 0);
        return;
    }

    int loop_count = 0;
    while (gs_effect_loop(default_effect, technique_name)) {
        loop_count++;
        gs_draw_sprite(NULL, 0, filter->width, filter->height);
    }
    if (loop_count == 0) {
        blog(LOG_WARNING, "[TRT Filter] gs_effect_loop() returned 0 iterations with technique '%s'", technique_name);
    }

    gs_blend_state_pop();

    if (!filter->debug_force_solid_color && image_param) {
        gs_effect_set_texture(image_param, nullptr);
    }

    // NOTE: Do NOT call obs_source_process_filter_end here — it redraws the upstream source
    // over our already-drawn output, completely overwriting the TRT-processed frame.
    
    // Step 8: Metrics pass (if enabled and interval reached)
    if (filter->metrics_enabled && filter->metrics_shader) {
        filter->metrics_frame_counter++;
        
        if (filter->metrics_frame_counter >= filter->metrics_interval) {
            filter->metrics_frame_counter = 0;
            
            // Barrier: Ensure postprocess is complete before metrics read the texture
            ID3D11Query *metricsQuery = nullptr;
            D3D11_QUERY_DESC metricsQueryDesc = {};
            metricsQueryDesc.Query = D3D11_QUERY_EVENT;
            if (SUCCEEDED(filter->d3d11_device->CreateQuery(&metricsQueryDesc, &metricsQuery))) {
                filter->d3d11_context->End(metricsQuery);
                filter->d3d11_context->Flush();
                BOOL metricsData = FALSE;
                int metrics_attempts = 0;
                const int max_metrics_attempts = 10000;
                while (filter->d3d11_context->GetData(metricsQuery, &metricsData, sizeof(BOOL), 0) == S_FALSE && metrics_attempts < max_metrics_attempts) {
                    Sleep(0);
                    metrics_attempts++;
                }
                if (metrics_attempts >= max_metrics_attempts) {
                    blog(LOG_WARNING, "[TRT Filter] Metrics barrier query timed out");
                }
                metricsQuery->Release();
            }
            
            // Get SRVs for input and output textures.
            // Use GetDesc() for the input SRV format — same hazard fix as preprocess.
            ID3D11ShaderResourceView *input_srv_for_metrics  = nullptr;
            ID3D11ShaderResourceView *output_srv_for_metrics = nullptr;

            ID3D11Texture2D *input_d3d11_for_metrics  = (ID3D11Texture2D *)gs_texture_get_obj(input_texture);
            ID3D11Texture2D *output_d3d11_for_metrics = filter->output_d3d11_texture;

            if (input_d3d11_for_metrics) {
                D3D11_TEXTURE2D_DESC in_desc = {};
                input_d3d11_for_metrics->GetDesc(&in_desc);
                D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
                srv_desc.Format = in_desc.Format;
                srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
                srv_desc.Texture2D.MipLevels = 1;
                srv_desc.Texture2D.MostDetailedMip = 0;
                filter->d3d11_device->CreateShaderResourceView(input_d3d11_for_metrics, &srv_desc, &input_srv_for_metrics);
            }

            {
                D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
                srv_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;  // output_d3d11_texture is always UNORM
                srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
                srv_desc.Texture2D.MipLevels = 1;
                srv_desc.Texture2D.MostDetailedMip = 0;
                filter->d3d11_device->CreateShaderResourceView(output_d3d11_for_metrics, &srv_desc, &output_srv_for_metrics);
            }
            
            // Create UAV for group sums buffer
            D3D11_UNORDERED_ACCESS_VIEW_DESC uav_desc = {};
            uav_desc.Format = DXGI_FORMAT_UNKNOWN;
            uav_desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
            uav_desc.Buffer.FirstElement = 0;
            uint32_t num_groups = ((filter->metrics_sample_size + 15) / 16) * ((filter->metrics_sample_size + 15) / 16);
            uav_desc.Buffer.NumElements = num_groups;
            uav_desc.Buffer.Flags = 0;
            
            ID3D11UnorderedAccessView *group_sums_uav = nullptr;
            filter->d3d11_device->CreateUnorderedAccessView(filter->metrics_staging_buffer, &uav_desc, &group_sums_uav);
            
            // Clear group sums buffer
            float clear_val[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            filter->d3d11_context->ClearUnorderedAccessViewFloat(group_sums_uav, clear_val);
            
            // Update constant buffer
            struct {
                uint32_t sample_size[2];
                uint32_t input_size[2];
            } metrics_constants;
            
            metrics_constants.sample_size[0] = filter->metrics_sample_size;
            metrics_constants.sample_size[1] = filter->metrics_sample_size;
            metrics_constants.input_size[0] = filter->width;
            metrics_constants.input_size[1] = filter->height;
            
            D3D11_MAPPED_SUBRESOURCE mapped;
            filter->d3d11_context->Map(filter->metrics_constant_buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
            memcpy(mapped.pData, &metrics_constants, sizeof(metrics_constants));
            filter->d3d11_context->Unmap(filter->metrics_constant_buffer, 0);
            
            // Clear OM RT bindings to prevent D3D11 hazard detection from nulling CS SRVs.
            filter->d3d11_context->OMSetRenderTargets(0, nullptr, nullptr);

            // Dispatch metrics shader
            filter->d3d11_context->CSSetShader(filter->metrics_shader, nullptr, 0);
            filter->d3d11_context->CSSetConstantBuffers(0, 1, &filter->metrics_constant_buffer);
            filter->d3d11_context->CSSetShaderResources(0, 1, &input_srv_for_metrics);
            filter->d3d11_context->CSSetShaderResources(1, 1, &output_srv_for_metrics);
            filter->d3d11_context->CSSetUnorderedAccessViews(0, 1, &group_sums_uav, nullptr);
            
            uint32_t groups_x = (filter->metrics_sample_size + 15) / 16;
            uint32_t groups_y = (filter->metrics_sample_size + 15) / 16;
            filter->d3d11_context->Dispatch(groups_x, groups_y, 1);
            
            // Unbind
            ID3D11ShaderResourceView *null_srv = nullptr;
            ID3D11UnorderedAccessView *null_uav = nullptr;
            filter->d3d11_context->CSSetShaderResources(0, 1, &null_srv);
            filter->d3d11_context->CSSetShaderResources(1, 1, &null_srv);
            filter->d3d11_context->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
            
            // Staging buffer for CPU readback — must exactly match GPU buffer description
            // (same MiscFlags, StructureByteStride, ByteWidth) for CopyResource to succeed.
            D3D11_BUFFER_DESC staging_desc = {};
            staging_desc.Usage             = D3D11_USAGE_STAGING;
            staging_desc.BindFlags         = 0;
            staging_desc.CPUAccessFlags    = D3D11_CPU_ACCESS_READ;
            staging_desc.MiscFlags         = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
            staging_desc.StructureByteStride = sizeof(float) * 4;  // float4 per group
            staging_desc.ByteWidth         = 64 * sizeof(float) * 4;  // 1024 bytes

            ID3D11Buffer *staging_buffer = nullptr;
            filter->d3d11_device->CreateBuffer(&staging_desc, nullptr, &staging_buffer);

            filter->d3d11_context->CopyResource(staging_buffer, filter->metrics_staging_buffer);

            D3D11_MAPPED_SUBRESOURCE mapped_staging;
            filter->d3d11_context->Map(staging_buffer, 0, D3D11_MAP_READ, 0, &mapped_staging);

            // Accumulate per-group float4: (out_lap, in_lap, out_lum, in_lum)
            const float *data = (const float *)mapped_staging.pData;
            float sum_out_lap = 0.0f, sum_in_lap = 0.0f;
            float sum_out_lum = 0.0f, sum_in_lum = 0.0f;
            for (uint32_t i = 0; i < num_groups; i++) {
                sum_out_lap += data[i * 4 + 0];
                sum_in_lap  += data[i * 4 + 1];
                sum_out_lum += data[i * 4 + 2];
                sum_in_lum  += data[i * 4 + 3];
            }

            filter->d3d11_context->Unmap(staging_buffer, 0);
            staging_buffer->Release();

            // Compute final metrics
            const float N = (float)(filter->metrics_sample_size * filter->metrics_sample_size);
            float out_sharpness   = sum_out_lap / N;
            float in_sharpness    = sum_in_lap  / N;
            float sharpness_delta = out_sharpness - in_sharpness;
            float out_mean_lum    = sum_out_lum  / N;

            // Temporal flicker: frame-to-frame change in output mean luminance.
            // Large values on a static scene indicate SR flickering artifacts.
            static float s_prev_out_lum = 0.0f;
            static bool  s_has_prev_lum = false;
            float flicker = s_has_prev_lum ? fabsf(out_mean_lum - s_prev_out_lum) : 0.0f;
            s_prev_out_lum = out_mean_lum;
            s_has_prev_lum = true;

            blog(LOG_INFO,
                 "[TRT Filter] Metrics (frame %llu): "
                 "out_sharp=%.5f  in_sharp=%.5f  sharp_delta=%+.5f  "
                 "out_lum=%.4f  flicker=%.5f  sample=%ux%u",
                 (unsigned long long)filter->inference_frame_count,
                 out_sharpness, in_sharpness, sharpness_delta,
                 out_mean_lum, flicker,
                 filter->metrics_sample_size, filter->metrics_sample_size);
            
            // Cleanup
            if (input_srv_for_metrics) {
                input_srv_for_metrics->Release();
            }
            if (output_srv_for_metrics) {
                output_srv_for_metrics->Release();
            }
            if (group_sums_uav) {
                group_sums_uav->Release();
            }
        }
    }
}

/* Source info */
static struct obs_source_info trt_filter_info;
static void init_trt_filter_info() {
    trt_filter_info.id = "obs_tensorrt_inference_filter";
    trt_filter_info.type = OBS_SOURCE_TYPE_FILTER;
    trt_filter_info.output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_SRGB;
    trt_filter_info.get_name = trt_filter_name;
    trt_filter_info.create = trt_filter_create;
    trt_filter_info.destroy = trt_filter_destroy;
    trt_filter_info.get_properties = trt_filter_properties;
    trt_filter_info.get_defaults = trt_filter_defaults;
    trt_filter_info.update = trt_filter_update;
    trt_filter_info.video_tick = trt_filter_tick;
    trt_filter_info.video_render = trt_filter_render;
}

/* Module entry */
OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("obs-tensorrt-filter", "en-US")

bool obs_module_load(void)
{
    blog(LOG_INFO, "TensorRT  Real-ESRGAN inference filter plugin 8.0 loaded (version %s)", PLUGIN_VERSION);
    init_trt_filter_info();
    obs_register_source(&trt_filter_info);
    blog(LOG_INFO, "registering source (version %s)", PLUGIN_VERSION);

    return true;
}
