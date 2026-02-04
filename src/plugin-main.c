#include <obs-module.h>
#include <graphics/graphics.h>
#include <util/platform.h>
#include <stdbool.h>
#include <d3d11.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include "cuda-kernel.h"

#define PLUGIN_VERSION "1.0.0"

/* 1.  per-instance data  */
struct cuda_filter_data {
    obs_source_t *context;
    
    // Graphics resources
    gs_texrender_t *render_unorm;
    gs_texture_t *output_texture;
    
    // Dimensions
    uint32_t width;
    uint32_t height;
    bool resources_allocated;
    bool initial_render;
    bool target_valid;
    
    // CUDA resources
    CUcontext cuda_context;
    CUstream cuda_stream;
    CUgraphicsResource cuda_input_resource;
    CUgraphicsResource cuda_output_resource;
    CUmodule cuda_module;
    CUfunction cuda_kernel;
    
    // CUDA device memory (for array access)
    CUarray cuda_input_array;
    CUarray cuda_output_array;
};

/* 2.  display name  */
static const char *cuda_filter_name(void *unused)
{
    UNUSED_PARAMETER(unused);
    return obs_module_text("CUDAPixelInvertFilter");
}

/* Helper: Check CUDA errors */
#define CUDA_CHECK(call) \
    do { \
        CUresult err = call; \
        if (err != CUDA_SUCCESS) { \
            const char *err_str; \
            cuGetErrorString(err, &err_str); \
            blog(LOG_ERROR, "[CUDA Filter] CUDA error at %s:%d - %s", __FILE__, __LINE__, err_str); \
            return false; \
        } \
    } while(0)

/* Helper: Check CUDA runtime errors */
#define CUDA_RT_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            blog(LOG_ERROR, "[CUDA Filter] CUDA runtime error at %s:%d - %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return false; \
        } \
    } while(0)

/* 3.  Create/recreate graphics resources */
static bool create_graphics_resources(struct cuda_filter_data *filter)
{
    if (filter->render_unorm) {
        gs_texrender_destroy(filter->render_unorm);
        filter->render_unorm = NULL;
    }
    
    if (filter->output_texture) {
        gs_texture_destroy(filter->output_texture);
        filter->output_texture = NULL;
    }
    
    // Create texrender for input (BGRA_UNORM, SRGB)
    filter->render_unorm = gs_texrender_create(GS_BGRA_UNORM, GS_ZS_NONE);
    if (!filter->render_unorm) {
        blog(LOG_ERROR, "[CUDA Filter] Failed to create render_unorm texrender");
        return false;
    }
    
    // Create output texture (BGRA_UNORM)
    filter->output_texture = gs_texture_create(
        filter->width, 
        filter->height, 
        GS_BGRA_UNORM, 
        1, 
        NULL, 
        0);
    if (!filter->output_texture) {
        blog(LOG_ERROR, "[CUDA Filter] Failed to create output texture");
        gs_texrender_destroy(filter->render_unorm);
        filter->render_unorm = NULL;
        return false;
    }
    
    return true;
}

/* 4.  Initialize CUDA context and resources */
static bool init_cuda_resources(struct cuda_filter_data *filter)
{
    // Check device type
    if (gs_get_device_type() != GS_DEVICE_DIRECT3D_11) {
        blog(LOG_ERROR, "[CUDA Filter] This filter requires D3D11 backend");
        return false;
    }
    
    // Get D3D11 device
    ID3D11Device *d3d11_device = (ID3D11Device *)gs_get_device_obj();
    if (!d3d11_device) {
        blog(LOG_ERROR, "[CUDA Filter] Failed to get D3D11 device");
        return false;
    }
    
    // Create CUDA context from D3D11 device
    CUDA_CHECK(cuD3D11CtxCreate(&filter->cuda_context, CU_CTX_SCHED_AUTO, d3d11_device));
    
    // Set CUDA context
    CUDA_CHECK(cuCtxSetCurrent(filter->cuda_context));
    
    // Create CUDA stream
    CUDA_CHECK(cuStreamCreate(&filter->cuda_stream, CU_STREAM_DEFAULT));
    
    // Get textures
    gs_texture_t *input_texture = gs_texrender_get_texture(filter->render_unorm);
    ID3D11Texture2D *input_d3d11 = (ID3D11Texture2D *)gs_texture_get_obj(input_texture);
    ID3D11Texture2D *output_d3d11 = (ID3D11Texture2D *)gs_texture_get_obj(filter->output_texture);
    
    if (!input_d3d11 || !output_d3d11) {
        blog(LOG_ERROR, "[CUDA Filter] Failed to get D3D11 textures");
        return false;
    }
    
    // Register input texture (read-only)
    CUDA_CHECK(cuGraphicsD3D11RegisterResource(
        &filter->cuda_input_resource,
        input_d3d11,
        CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY));
    
    // Register output texture (write-discard)
    CUDA_CHECK(cuGraphicsD3D11RegisterResource(
        &filter->cuda_output_resource,
        output_d3d11,
        CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));
    
    // Load CUDA kernel (we'll compile it at runtime or use PTX)
    // For now, we'll use a simple approach with a compiled kernel
    // In a real implementation, you'd load a .cubin or .ptx file
    blog(LOG_INFO, "[CUDA Filter] CUDA resources initialized");
    
    return true;
}

/* 5.  Process frame with CUDA */
static bool process_frame_cuda(struct cuda_filter_data *filter)
{
    CUresult cu_res;
    
    // Map resources
    CUgraphicsResource resources[2] = {
        filter->cuda_input_resource,
        filter->cuda_output_resource
    };
    
    CUDA_CHECK(cuGraphicsMapResources(2, resources, filter->cuda_stream));
    
    // Get mapped arrays
    cu_res = cuGraphicsSubResourceGetMappedArray(
        &filter->cuda_input_array,
        filter->cuda_input_resource,
        0, 0);
    if (cu_res != CUDA_SUCCESS) {
        blog(LOG_ERROR, "[CUDA Filter] Failed to get mapped input array");
        cuGraphicsUnmapResources(2, resources, filter->cuda_stream);
        return false;
    }
    
    cu_res = cuGraphicsSubResourceGetMappedArray(
        &filter->cuda_output_array,
        filter->cuda_output_resource,
        0, 0);
    if (cu_res != CUDA_SUCCESS) {
        blog(LOG_ERROR, "[CUDA Filter] Failed to get mapped output array");
        cuGraphicsUnmapResources(2, resources, filter->cuda_stream);
        return false;
    }
    
    // Copy from input array to linear memory, process, copy back
    // For simplicity, we'll use cuMemcpy2D to copy array to device memory
    // Then launch kernel, then copy back
    
    // Allocate temporary device memory
    size_t image_size = filter->width * filter->height * 4; // BGRA
    
    unsigned char *d_input, *d_output;
    CUDA_RT_CHECK(cudaMalloc(&d_input, image_size));
    CUDA_RT_CHECK(cudaMalloc(&d_output, image_size));
    
    // Copy from array to linear memory
    CUDA_MEMCPY2D copy_params = {0};
    copy_params.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy_params.srcArray = filter->cuda_input_array;
    copy_params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copy_params.dstDevice = (CUdeviceptr)d_input;
    copy_params.dstPitch = filter->width * 4;
    copy_params.WidthInBytes = filter->width * 4;
    copy_params.Height = filter->height;
    
    CUDA_CHECK(cuMemcpy2D(&copy_params));
    
    // Convert CUstream to cudaStream_t for kernel launch
    // CUstream and cudaStream_t are compatible types (both are void*)
    cudaStream_t runtime_stream = (cudaStream_t)filter->cuda_stream;
    
    // Launch CUDA kernel to process the image
    cudaError_t kernel_err = launch_invert_kernel(
        d_input,
        d_output,
        (int)filter->width,
        (int)filter->height,
        runtime_stream);
    
    if (kernel_err != cudaSuccess) {
        blog(LOG_ERROR, "[CUDA Filter] Kernel launch failed: %s", cudaGetErrorString(kernel_err));
        CUDA_RT_CHECK(cudaFree(d_input));
        CUDA_RT_CHECK(cudaFree(d_output));
        cuGraphicsUnmapResources(2, resources, filter->cuda_stream);
        return false;
    }
    
    // Copy back to output array
    copy_params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy_params.srcDevice = (CUdeviceptr)d_output;
    copy_params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copy_params.dstArray = filter->cuda_output_array;
    
    CUDA_CHECK(cuMemcpy2D(&copy_params));
    
    // Free temporary memory
    CUDA_RT_CHECK(cudaFree(d_input));
    CUDA_RT_CHECK(cudaFree(d_output));
    
    // Unmap resources
    CUDA_CHECK(cuGraphicsUnmapResources(2, resources, filter->cuda_stream));
    
    // Synchronize stream
    CUDA_CHECK(cuStreamSynchronize(filter->cuda_stream));
    
    return true;
}

/* 6.  Cleanup CUDA resources */
static void cleanup_cuda_resources(struct cuda_filter_data *filter)
{
    // Set CUDA context before cleanup
    if (filter->cuda_context) {
        cuCtxSetCurrent(filter->cuda_context);
    }
    
    if (filter->cuda_input_resource) {
        cuGraphicsUnregisterResource(filter->cuda_input_resource);
        filter->cuda_input_resource = NULL;
    }
    
    if (filter->cuda_output_resource) {
        cuGraphicsUnregisterResource(filter->cuda_output_resource);
        filter->cuda_output_resource = NULL;
    }
    
    if (filter->cuda_stream) {
        cuStreamDestroy(filter->cuda_stream);
        filter->cuda_stream = NULL;
    }
    
    if (filter->cuda_context) {
        cuCtxDestroy(filter->cuda_context);
        filter->cuda_context = NULL;
    }
}

/* 7.  destructor  */
static void cuda_filter_destroy(void *data)
{
    struct cuda_filter_data *filter = data;
    
    obs_enter_graphics();
    
    // Cleanup CUDA resources
    cleanup_cuda_resources(filter);
    
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

/* 8.  constructor  */
static void *cuda_filter_create(obs_data_t *settings, obs_source_t *source)
{
    UNUSED_PARAMETER(settings);
    
    struct cuda_filter_data *filter = bzalloc(sizeof(*filter));
    filter->context = source;
    filter->width = 0;
    filter->height = 0;
    filter->resources_allocated = false;
    filter->initial_render = false;
    filter->target_valid = false;
    filter->render_unorm = NULL;
    filter->output_texture = NULL;
    filter->cuda_context = NULL;
    filter->cuda_stream = NULL;
    filter->cuda_input_resource = NULL;
    filter->cuda_output_resource = NULL;
    filter->cuda_module = NULL;
    filter->cuda_kernel = NULL;
    
    // Initialize CUDA driver API
    CUresult cu_res = cuInit(0);
    if (cu_res != CUDA_SUCCESS) {
        blog(LOG_ERROR, "[CUDA Filter] Failed to initialize CUDA driver");
        bfree(filter);
        return NULL;
    }
    
    return filter;
}

/* 9.  tick function - track size changes */
static void cuda_filter_tick(void *data, float seconds)
{
    UNUSED_PARAMETER(seconds);
    
    struct cuda_filter_data *filter = data;
    
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
        if (create_graphics_resources(filter)) {
            filter->resources_allocated = true;
            filter->initial_render = false;
        } else {
            filter->resources_allocated = false;
        }
        obs_leave_graphics();
    }
}

/* 10.  per-frame render  */
static void cuda_filter_render(void *data, gs_effect_t *effect)
{
    UNUSED_PARAMETER(effect);
    
    struct cuda_filter_data *filter = data;
    
    if (!filter->target_valid || !filter->resources_allocated) {
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
    
    // Step 2: Initialize CUDA resources on first render after allocation
    if (!filter->initial_render) {
        if (!init_cuda_resources(filter)) {
            blog(LOG_ERROR, "[CUDA Filter] Failed to initialize CUDA resources");
            obs_source_skip_video_filter(filter->context);
            return;
        }
        filter->initial_render = true;
    }
    
    // Step 3: Process frame with CUDA
    if (!process_frame_cuda(filter)) {
        blog(LOG_ERROR, "[CUDA Filter] Failed to process frame with CUDA");
        obs_source_skip_video_filter(filter->context);
        return;
    }
    
    // Step 4: Draw output texture
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

/* 11.  source info  */
static struct obs_source_info cuda_filter_info = {
    .id           = "obs_cuda_pixel_invert_filter",
    .type         = OBS_SOURCE_TYPE_FILTER,
    .output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_SRGB,
    .get_name     = cuda_filter_name,
    .create       = cuda_filter_create,
    .destroy      = cuda_filter_destroy,
    .video_tick   = cuda_filter_tick,
    .video_render = cuda_filter_render,
};

/* 7.  module entry  */
OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("obs-grayscale-filter", "en-US")

bool obs_module_load(void)
{
    blog(LOG_INFO, "CUDA pixel invert filter plugin loaded (version %s)", PLUGIN_VERSION);
    obs_register_source(&cuda_filter_info);
    blog(LOG_INFO, "registering source (version %s)", PLUGIN_VERSION);

    return true;
}