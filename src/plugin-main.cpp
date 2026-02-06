// NOTE: This file was converted from C to C++ to support CUDA/TensorRT integration.
// Original content was in plugin-main.c.

#include <obs-module.h>
#include <graphics/graphics.h>
#include <util/platform.h>
#include <stdbool.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <windows.h>
#include "trt_runner.h"

#define PLUGIN_VERSION "1.0.0"

// Engine file path (can be made configurable)
#define DEFAULT_ENGINE_PATH "identity_cnn_256_fp16.engine"

/* Per-instance data */
struct trt_filter_data {
    obs_source_t *context;
    
    // Graphics resources
    gs_texrender_t *render_unorm;
    gs_texture_t *output_texture;
    
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
    
    // D3D11 buffers for TRT input/output (FP16 NCHW 1x3x256x256)
    ID3D11Buffer *trt_input_buffer;   // Structured buffer for FP16 input
    ID3D11Buffer *trt_output_buffer;  // Structured buffer for FP16 output
    ID3D11UnorderedAccessView *trt_input_uav;
    ID3D11UnorderedAccessView *trt_output_uav;
    ID3D11ShaderResourceView *trt_output_srv;
    
    // Constant buffer for shaders
    ID3D11Buffer *constant_buffer;
    
    // CUDA resources (runtime API)
    cudaStream_t cuda_stream;
    cudaGraphicsResource_t cuda_trt_input_resource;
    cudaGraphicsResource_t cuda_trt_output_resource;
    void *cuda_trt_input_ptr;
    void *cuda_trt_output_ptr;
    
    // TensorRT runner
    struct trt_runner trt_runner;
    bool trt_initialized;
    
    // D3D11 fence for synchronization (currently unused, placeholder for future sync)
    ID3D11Query *d3d11_fence;
    // ID3D11Fence *d3d11_fence_obj;
    HANDLE shared_handle;
    UINT64 fence_value;
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
    
    // Create texrender for input (BGRA_UNORM, SRGB)
    filter->render_unorm = gs_texrender_create(GS_BGRA_UNORM, GS_ZS_NONE);
    if (!filter->render_unorm) {
        blog(LOG_ERROR, "[TRT Filter] Failed to create render_unorm texrender");
        return false;
    }
    
    // Create output texture (BGRA_UNORM)
    filter->output_texture = gs_texture_create(
        filter->width, 
        filter->height, 
        GS_BGRA_UNORM, 
        1, 
        nullptr, 
        0);
    if (!filter->output_texture) {
        blog(LOG_ERROR, "[TRT Filter] Failed to create output texture");
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
    
    // TRT expects FP16 NCHW 1x3x256x256
    const uint32_t tensor_size = 1 * 3 * 256 * 256;
    
    // Create structured buffer for input (RWStructuredBuffer<uint> in shader, stores FP16 in low 16 bits)
    D3D11_BUFFER_DESC buffer_desc = {};
    buffer_desc.Usage = D3D11_USAGE_DEFAULT;
    buffer_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    buffer_desc.CPUAccessFlags = 0;
    buffer_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    buffer_desc.StructureByteStride = sizeof(uint32_t); // we store FP16 in lower 16 bits of uint
    buffer_desc.ByteWidth = tensor_size * sizeof(uint32_t);
    
    HRESULT hr = device->CreateBuffer(&buffer_desc, nullptr, &filter->trt_input_buffer);
    D3D11_CHECK(hr, "Failed to create TRT input buffer");
    
    hr = device->CreateBuffer(&buffer_desc, nullptr, &filter->trt_output_buffer);
    D3D11_CHECK(hr, "Failed to create TRT output buffer");
    
    // Create UAVs
    D3D11_UNORDERED_ACCESS_VIEW_DESC uav_desc = {};
    uav_desc.Format = DXGI_FORMAT_R32_UINT;
    uav_desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    uav_desc.Buffer.FirstElement = 0;
    uav_desc.Buffer.NumElements = tensor_size;
    
    hr = device->CreateUnorderedAccessView(filter->trt_input_buffer, &uav_desc, &filter->trt_input_uav);
    D3D11_CHECK(hr, "Failed to create TRT input UAV");
    
    hr = device->CreateUnorderedAccessView(filter->trt_output_buffer, &uav_desc, &filter->trt_output_uav);
    D3D11_CHECK(hr, "Failed to create TRT output UAV");
    
    // Create SRV for output buffer (for postprocess shader)
    D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = DXGI_FORMAT_R32_UINT;
    srv_desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
    srv_desc.Buffer.FirstElement = 0;
    srv_desc.Buffer.NumElements = tensor_size;
    
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
    CUDA_CHECK(cudaD3D11SetDirect3DDevice(filter->d3d11_device));
    
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
    
    // Use runtime stream directly (no conversion needed)
    bool success = trt_runner_init(&filter->trt_runner, engine_path, filter->cuda_stream);
    bfree(engine_path);
    
    if (!success) {
        blog(LOG_ERROR, "[TRT Filter] Failed to initialize TensorRT");
        return false;
    }
    
    filter->trt_initialized = true;
    blog(LOG_INFO, "[TRT Filter] TensorRT initialized");
    
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

/* Cleanup D3D11 resources */
static void cleanup_d3d11_resources(struct trt_filter_data *filter)
{
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
    
    obs_leave_graphics();
    
    bfree(filter);
}

/* Constructor */
static void *trt_filter_create(obs_data_t *settings, obs_source_t *source)
{
    UNUSED_PARAMETER(settings);
    
    auto *filter = static_cast<trt_filter_data *>(bzalloc(sizeof(trt_filter_data)));
    filter->context = source;
    filter->width = 0;
    filter->height = 0;
    filter->resources_allocated = false;
    filter->target_valid = false;
    filter->trt_initialized = false;
    
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
        
        // Create TRT buffers (only once, size is fixed at 256x256)
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
        obs_leave_graphics();
    }
}

/* Per-frame render */
static void trt_filter_render(void *data, gs_effect_t *effect)
{
    UNUSED_PARAMETER(effect);
    
    auto *filter = static_cast<trt_filter_data *>(data);
    
    if (!filter->target_valid || !filter->resources_allocated || !filter->trt_initialized) {
        obs_source_skip_video_filter(filter->context);
        return;
    }
    
    obs_source_t *target = obs_filter_get_target(filter->context);
    obs_source_t *parent = obs_filter_get_parent(filter->context);
    
    if (!target || !parent) {
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
    
    // Create SRV for input texture
    ID3D11ShaderResourceView *input_srv = nullptr;
    D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srv_desc.Texture2D.MipLevels = 1;
    srv_desc.Texture2D.MostDetailedMip = 0;
    filter->d3d11_device->CreateShaderResourceView(input_d3d11, &srv_desc, &input_srv);
    
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
    
    // Calculate scale and offset for centered crop
    float scale_x = (float)filter->width / 256.0f;
    float scale_y = (float)filter->height / 256.0f;
    float scale = (scale_x > scale_y) ? scale_x : scale_y;
    constants.scale[0] = scale;
    constants.scale[1] = scale;
    constants.offset[0] = (filter->width - 256.0f * scale) * 0.5f;
    constants.offset[1] = (filter->height - 256.0f * scale) * 0.5f;
    
    D3D11_MAPPED_SUBRESOURCE mapped;
    filter->d3d11_context->Map(filter->constant_buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    memcpy(mapped.pData, &constants, sizeof(constants));
    filter->d3d11_context->Unmap(filter->constant_buffer, 0);
    
    // Set compute shader state
    filter->d3d11_context->CSSetShader(filter->preprocess_shader, nullptr, 0);
    filter->d3d11_context->CSSetConstantBuffers(0, 1, &filter->constant_buffer);
    filter->d3d11_context->CSSetShaderResources(0, 1, &input_srv);
    filter->d3d11_context->CSSetUnorderedAccessViews(0, 1, &filter->trt_input_uav, nullptr);
    
    // Dispatch compute shader (256x256 = 16x16 thread groups)
    filter->d3d11_context->Dispatch(16, 16, 1);
    
    // Unbind resources
    ID3D11ShaderResourceView *null_srv = nullptr;
    ID3D11UnorderedAccessView *null_uav = nullptr;
    filter->d3d11_context->CSSetShaderResources(0, 1, &null_srv);
    filter->d3d11_context->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
    
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
    
    // Step 4: Run TensorRT inference
    trt_runner_set_buffers(&filter->trt_runner, filter->cuda_trt_input_ptr, filter->cuda_trt_output_ptr);
    if (!trt_runner_enqueue(&filter->trt_runner)) {
        blog(LOG_ERROR, "[TRT Filter] TensorRT inference failed");
        cudaGraphicsUnmapResources(2, resources, filter->cuda_stream);
        obs_source_skip_video_filter(filter->context);
        return;
    }
    
    // Step 5: Synchronize CUDA -> D3D11
    CUDA_CHECK_VOID(cudaStreamSynchronize(filter->cuda_stream));
    
    // Unmap CUDA resources
    CUDA_CHECK_VOID(cudaGraphicsUnmapResources(2, resources, filter->cuda_stream));
    
    // Step 6: Postprocess with compute shader (FP16 NCHW -> BGRA)
    ID3D11Texture2D *output_d3d11 = (ID3D11Texture2D *)gs_texture_get_obj(filter->output_texture);
    
    // Create UAV for output texture
    ID3D11UnorderedAccessView *output_uav = nullptr;
    D3D11_UNORDERED_ACCESS_VIEW_DESC uav_desc = {};
    uav_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    uav_desc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
    uav_desc.Texture2D.MipSlice = 0;
    
    HRESULT hr = filter->d3d11_device->CreateUnorderedAccessView(output_d3d11, &uav_desc, &output_uav);
    if (FAILED(hr)) {
        blog(LOG_ERROR, "[TRT Filter] Failed to create output UAV");
        obs_source_skip_video_filter(filter->context);
        return;
    }
    
    // Update constant buffer for postprocess
    struct {
        uint32_t output_size[2];
        uint32_t tensor_size[2];
    } post_constants;
    
    post_constants.output_size[0] = filter->width;
    post_constants.output_size[1] = filter->height;
    post_constants.tensor_size[0] = 256;
    post_constants.tensor_size[1] = 256;
    
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
    filter->d3d11_context->Dispatch(groups_x, groups_y, 1);
    
    // Unbind resources
    filter->d3d11_context->CSSetShaderResources(0, 1, &null_srv);
    filter->d3d11_context->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
    
    if (output_uav) {
        output_uav->Release();
    }
    
    // Flush D3D11 commands
    filter->d3d11_context->Flush();
    
    // Step 7: Draw output texture
    if (!obs_source_process_filter_begin(filter->context, GS_BGRA_UNORM, OBS_NO_DIRECT_RENDERING)) {
        return;
    }
    
    gs_effect_t *default_effect = obs_get_base_effect(OBS_EFFECT_DEFAULT);
    gs_eparam_t *image_param = gs_effect_get_param_by_name(default_effect, "image");
    
    gs_effect_set_texture(image_param, filter->output_texture);
    
    gs_blend_state_push();
    gs_blend_function(GS_BLEND_ONE, GS_BLEND_INVSRCALPHA);
    
    while (gs_effect_loop(default_effect, "Draw")) {
        gs_draw_sprite(filter->output_texture, 0, filter->width, filter->height);
    }
    
    gs_blend_state_pop();
    
    obs_source_process_filter_end(filter->context, default_effect, 0, 0);
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
    trt_filter_info.video_tick = trt_filter_tick;
    trt_filter_info.video_render = trt_filter_render;
}

/* Module entry */
OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("obs-tensorrt-filter", "en-US")

bool obs_module_load(void)
{
    blog(LOG_INFO, "TensorRT inference filter plugin loaded (version %s)", PLUGIN_VERSION);
    init_trt_filter_info();
    obs_register_source(&trt_filter_info);
    blog(LOG_INFO, "registering source (version %s)", PLUGIN_VERSION);

    return true;
}
