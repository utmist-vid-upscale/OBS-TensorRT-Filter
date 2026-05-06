# TensorRT Inference Filter Implementation

## Overview

This implementation integrates TensorRT inference into an OBS filter plugin using D3D11 HLSL compute shaders for all pre/postprocessing. The filter processes video frames in real-time using the NVIDIA VideoFX pattern. Here is a demo video: https://youtu.be/E74Dqn3iE-U

## Architecture

### Pipeline Flow

1. **Frame Capture**: Render upstream source to `gs_texrender_t` with `GS_BGRA_UNORM` and `GS_CS_SRGB`
2. **Preprocessing**: D3D11 compute shader converts BGRA → FP16 NCHW (1x3x256x256)
3. **TensorRT Inference**: Runs on CUDA stream with proper D3D11↔CUDA synchronization
4. **Postprocessing**: D3D11 compute shader converts FP16 NCHW → BGRA8 UNORM
5. **Output**: Draw processed texture to screen

### Key Components

#### Files Created/Modified

1. **`src/trt_runner.h` / `src/trt_runner.cpp`**
   - TensorRT engine loading and inference wrapper
   - Handles engine deserialization, context creation, and `enqueueV3` calls
   - Manages tensor names and buffer addresses

2. **`data/preprocess.hlsl`**
   - D3D11 compute shader (cs_5_0)
   - Reads BGRA UNORM texture
   - Resizes/crops to 256x256 (centered crop)
   - Converts to RGB, normalizes to [0,1]
   - Writes FP16 values to structured buffer in NCHW layout

3. **`data/postprocess.hlsl`**
   - D3D11 compute shader (cs_5_0)
   - Reads FP16 NCHW tensor from structured buffer
   - Converts to BGRA8 UNORM
   - Writes to output texture via UAV

4. **`src/plugin-main.c`** (completely rewritten)
   - Main filter implementation
   - Manages all graphics, D3D11, CUDA, and TensorRT resources
   - Handles frame processing pipeline

5. **`CMakeLists.txt`** (updated)
   - Added TensorRT library linking
   - Added `trt_runner.cpp` to sources
   - Added `d3dcompiler` library for shader compilation

## Resource Ownership and Lifecycle

### Graphics Resources (OBS API)

- **`gs_texrender_t *render_unorm`**: Input frame capture (BGRA_UNORM, SRGB)
- **`gs_texture_t *output_texture`**: Final output texture (BGRA_UNORM)
- **Ownership**: Created in `create_graphics_resources()`, destroyed in `trt_filter_destroy()`
- **Graphics Context**: All `gs_*` calls must be within `obs_enter_graphics()` / `obs_leave_graphics()`

### D3D11 Resources

- **`ID3D11Device *d3d11_device`**: Retrieved from OBS graphics API
- **`ID3D11DeviceContext *d3d11_context`**: Immediate context for command submission
- **`ID3D11ComputeShader *preprocess_shader`**: Compiled from `preprocess.hlsl`
- **`ID3D11ComputeShader *postprocess_shader`**: Compiled from `postprocess.hlsl`
- **`ID3D11Buffer *trt_input_buffer`**: Structured buffer for FP16 input (1x3x256x256)
- **`ID3D11Buffer *trt_output_buffer`**: Structured buffer for FP16 output (1x3x256x256)
- **`ID3D11UnorderedAccessView *trt_input_uav`**: UAV for input buffer
- **`ID3D11UnorderedAccessView *trt_output_uav`**: UAV for output buffer
- **`ID3D11ShaderResourceView *trt_output_srv`**: SRV for output buffer (used by postprocess shader)
- **`ID3D11Buffer *constant_buffer`**: Constant buffer for shader parameters
- **Ownership**: Created in `create_trt_buffers()` / `load_compute_shaders()`, destroyed in `cleanup_d3d11_resources()`
- **Graphics Context**: D3D11 device/context access is safe within `obs_enter_graphics()` / `obs_leave_graphics()`

### CUDA Resources

- **`CUcontext cuda_context`**: Created from D3D11 device via `cuD3D11CtxCreate()`
- **`CUstream cuda_stream`**: CUDA stream for async operations
- **`CUgraphicsResource cuda_trt_input_resource`**: Registered D3D11 input buffer
- **`CUgraphicsResource cuda_trt_output_resource`**: Registered D3D11 output buffer
- **Ownership**: Created in `init_cuda_resources()`, destroyed in `cleanup_cuda_resources()`
- **Graphics Context**: CUDA-D3D11 interop requires graphics context. Resource mapping/unmapping happens in render function.

### TensorRT Resources

- **`struct trt_runner`**: Contains runtime, engine, context, and tensor info
- **Ownership**: Initialized in `init_tensorrt()`, destroyed in `trt_runner_destroy()`
- **Graphics Context**: TensorRT operations use CUDA context, which is tied to D3D11 device

## Graphics Context Requirements

### `obs_enter_graphics()` / `obs_leave_graphics()` Usage

**Required in:**

1. **`trt_filter_tick()`**
   - When creating/recreating graphics resources (`create_graphics_resources()`)
   - When creating D3D11 buffers (`create_trt_buffers()`)
   - When loading compute shaders (`load_compute_shaders()`)
   - When initializing CUDA resources (`init_cuda_resources()`)
   - When initializing TensorRT (`init_tensorrt()`)

2. **`trt_filter_destroy()`**
   - All cleanup operations (graphics, D3D11, CUDA, TensorRT)

**NOT required in:**

- **`trt_filter_render()`**: OBS automatically provides graphics context during render
- **`trt_filter_create()`**: No graphics resources created yet (only CUDA driver init)

### Synchronization Points

1. **D3D11 → CUDA**: After preprocessing compute shader
   - `d3d11_context->Flush()` to ensure D3D11 commands complete
   - `cuGraphicsMapResources()` to map D3D11 buffers to CUDA
   - `cuGraphicsResourceGetMappedPointer()` to get CUDA device pointers

2. **CUDA → D3D11**: After TensorRT inference
   - `cuStreamSynchronize()` to wait for CUDA operations
   - `cuGraphicsUnmapResources()` to unmap buffers

3. **D3D11 → OBS**: After postprocessing
   - `d3d11_context->Flush()` to ensure compute shader completes
   - OBS render pipeline handles final presentation

## Buffer Layout

### TensorRT Input/Output Buffers

- **Format**: Structured buffer of `uint16_t` (FP16 values)
- **Size**: `1 * 3 * 256 * 256 = 196,608` elements = `393,216` bytes
- **Layout**: NCHW
  - Channel 0 (R): `[0 * 256*256 + y * 256 + x]`
  - Channel 1 (G): `[1 * 256*256 + y * 256 + x]`
  - Channel 2 (B): `[2 * 256*256 + y * 256 + x]`

### Preprocessing

- Input: BGRA UNORM texture (variable size, e.g., 1920x1080)
- Output: FP16 NCHW buffer (256x256)
- Process: Centered crop with bilinear sampling, BGRA→RGB conversion, normalization

### Postprocessing

- Input: FP16 NCHW buffer (256x256)
- Output: BGRA UNORM texture (original size)
- Process: Read FP16 values, convert to float, clamp to [0,1], write BGRA

## Error Handling

All operations check for errors:
- CUDA: `CUDA_CHECK()` / `CUDA_RT_CHECK()` macros
- D3D11: `D3D11_CHECK()` macro
- TensorRT: Return values from `trt_runner_*` functions
- OBS: NULL checks and return value validation

On error, the filter calls `obs_source_skip_video_filter()` to pass through the original frame.

## Configuration

- **Engine file**: Default path is `"identity_cnn_256_fp16.engine"` (can be made configurable via settings)
- **Tensor size**: Fixed at 256x256 (hardcoded, can be made configurable)
- **Precision**: FP16 (hardcoded, matches engine)

## Build Requirements

- TensorRT 10.14+ (default path: `C:/TensorRT-10.14.1.48/`)
- CUDA Toolkit (for CUDA-D3D11 interop)
- D3D11 and D3DCompiler (Windows SDK)
- OBS Studio with D3D11 graphics backend

## Testing Checklist

- [ ] Engine file loads successfully
- [ ] Compute shaders compile without errors
- [ ] Resources created on first frame
- [ ] Resources recreated on size change
- [ ] Preprocessing produces correct FP16 values
- [ ] TensorRT inference runs successfully
- [ ] Postprocessing produces correct BGRA output
- [ ] No memory leaks on filter destruction
- [ ] Proper synchronization (no visual artifacts)
- [ ] Performance is acceptable for real-time use

## Known Limitations

1. Fixed tensor size (256x256) - input is cropped/scaled
2. Engine file path is hardcoded (should be configurable)
3. No error recovery if TensorRT fails mid-stream
4. Single engine per filter instance (no dynamic switching)

## Future Improvements

1. Make engine path configurable via OBS settings
2. Support variable tensor sizes
3. Add performance profiling hooks
4. Implement error recovery mechanisms
5. Support multiple engines with switching


# UTMIST Notes for Repo

## Quick-Start Build Guide (Windows, VS 2022)

### One-time setup
```bash
git clone https://github.com/utmist-vid-upscale/obs-grayscale-filter.git my-plugin
cd my-plugin
cmake --preset windows-x64   # downloads deps & generates VS solution
```
## 1. Build the DLL (any terminal)
```bash
cmake --build build_x64 --config Release --parallel
```

## 2. Install to OBS (admin **cmd.exe**)
```bash
cmake --install build_x64 --config Release --prefix "%ProgramData%\obs-studio\plugins"
```

# OBS Plugin Template

## Introduction

The plugin template is meant to be used as a starting point for OBS Studio plugin development. It includes:

* Boilerplate plugin source code
* A CMake project file
* GitHub Actions workflows and repository actions

## Supported Build Environments

| Platform  | Tool   |
|-----------|--------|
| Windows   | Visal Studio 17 2022 |
| macOS     | XCode 16.0 |
| Windows, macOS  | CMake 3.30.5 |
| Ubuntu 24.04 | CMake 3.28.3 |
| Ubuntu 24.04 | `ninja-build` |
| Ubuntu 24.04 | `pkg-config`
| Ubuntu 24.04 | `build-essential` |

## Quick Start

An absolute bare-bones [Quick Start Guide](https://github.com/obsproject/obs-plugintemplate/wiki/Quick-Start-Guide) is available in the wiki.

## Documentation

All documentation can be found in the [Plugin Template Wiki](https://github.com/obsproject/obs-plugintemplate/wiki).

Suggested reading to get up and running:

* [Getting started](https://github.com/obsproject/obs-plugintemplate/wiki/Getting-Started)
* [Build system requirements](https://github.com/obsproject/obs-plugintemplate/wiki/Build-System-Requirements)
* [Build system options](https://github.com/obsproject/obs-plugintemplate/wiki/CMake-Build-System-Options)

## GitHub Actions & CI

Default GitHub Actions workflows are available for the following repository actions:

* `push`: Run for commits or tags pushed to `master` or `main` branches.
* `pr-pull`: Run when a Pull Request has been pushed or synchronized.
* `dispatch`: Run when triggered by the workflow dispatch in GitHub's user interface.
* `build-project`: Builds the actual project and is triggered by other workflows.
* `check-format`: Checks CMake and plugin source code formatting and is triggered by other workflows.

The workflows make use of GitHub repository actions (contained in `.github/actions`) and build scripts (contained in `.github/scripts`) which are not needed for local development, but might need to be adjusted if additional/different steps are required to build the plugin.

### Retrieving build artifacts

Successful builds on GitHub Actions will produce build artifacts that can be downloaded for testing. These artifacts are commonly simple archives and will not contain package installers or installation programs.

### Building a Release

To create a release, an appropriately named tag needs to be pushed to the `main`/`master` branch using semantic versioning (e.g., `12.3.4`, `23.4.5-beta2`). A draft release will be created on the associated repository with generated installer packages or installation programs attached as release artifacts.

## Signing and Notarizing on macOS

Basic concepts of codesigning and notarization on macOS are explained in the correspodning [Wiki article](https://github.com/obsproject/obs-plugintemplate/wiki/Codesigning-On-macOS) which has a specific section for the [GitHub Actions setup](https://github.com/obsproject/obs-plugintemplate/wiki/Codesigning-On-macOS#setting-up-code-signing-for-github-actions).
