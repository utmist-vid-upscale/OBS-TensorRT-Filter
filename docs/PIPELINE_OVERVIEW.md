# Pipeline Technical Overview

This document describes the current working implementation of the OBS TensorRT super-resolution filter. For a history of bugs found and fixed during development, see [`PIPELINE_FIXES.md`](../PIPELINE_FIXES.md).

---

## Architecture

```
OBS source frame
      │
      ▼
[Step 1] gs_texrender_t (BGRA UNORM, GS_CS_SRGB context)
      │
      ▼ D3D11 compute — preprocess.hlsl
[Step 2] trt_input_buffer  (FP32 NCHW structured buffer, 256×256)
      │
      ▼ Flush + cudaGraphicsMapResources
[Step 3] CUDA device pointer to trt_input_buffer
      │
      ▼ TensorRT enqueueV3
[Step 4] CUDA device pointer to trt_output_buffer (FP32 NCHW, 1024×1024)
      │
      ▼ cudaStreamSynchronize + cudaGraphicsUnmapResources
[Step 5] trt_output_buffer returned to D3D11
      │
      ▼ D3D11 compute — postprocess.hlsl
[Step 6] output_d3d11_texture  (BGRA UNORM, W×H)
      │
      ▼ CopyResource
[Step 7] output_texture (gs_texture_t, BGRA UNORM)
      │
      ▼ obs_source_process_filter_begin + gs_draw_sprite
OBS rendered output
```

---

## Resource Types and Ownership

### OBS / Graphics resources
| Resource | Type | Created in | Destroyed in |
|----------|------|-----------|-------------|
| `render_unorm` | `gs_texrender_t` | `create_graphics_resources()` | `trt_filter_destroy()` |
| `output_texture` | `gs_texture_t` | `create_graphics_resources()` | `trt_filter_destroy()` |
| `output_d3d11_texture` | `ID3D11Texture2D` | `create_graphics_resources()` | `trt_filter_destroy()` |

### D3D11 pipeline resources
| Resource | Type | Notes |
|----------|------|-------|
| `trt_input_buffer` | `ID3D11Buffer` (structured, DEFAULT) | Stride = 4 bytes; 196 608 elements |
| `trt_output_buffer` | `ID3D11Buffer` (structured, DEFAULT) | Stride = 4 bytes; 3 145 728 elements |
| `trt_input_uav` | `ID3D11UnorderedAccessView` | Written by preprocess shader |
| `trt_output_srv` | `ID3D11ShaderResourceView` | Read by postprocess shader |
| `constant_buffer` | `ID3D11Buffer` (DYNAMIC) | Shared between preprocess and postprocess — see [Pitfall: shared cbuffer](#pitfall-shared-constant-buffer) |
| `linear_sampler` | `ID3D11SamplerState` | CLAMP, LINEAR; bound to CS s0 for preprocess |

### CUDA resources
| Resource | Notes |
|----------|-------|
| `cuda_stream` | All CUDA work submitted here |
| `cuda_trt_input_resource` | `cudaGraphicsD3D11RegisterResource` of `trt_input_buffer` |
| `cuda_trt_output_resource` | `cudaGraphicsD3D11RegisterResource` of `trt_output_buffer` |

### TensorRT resources
Managed by `struct trt_runner` in `trt_runner.cpp`. Holds `IRuntime`, `ICudaEngine`, `IExecutionContext`, tensor names, and byte sizes reported at init.

---

## Step-by-Step Per-Frame Flow

### Step 1 — Source capture
```cpp
gs_texrender_reset(render_unorm);
gs_texrender_begin_with_color_space(render_unorm, W, H, GS_CS_SRGB);
    obs_source_default_render(target);   // or obs_source_video_render
gs_texrender_end(render_unorm);
```
The texrender is created as `GS_BGRA_UNORM`. When rendered in `GS_CS_SRGB` context into a UNORM (not UNORM_SRGB) texture, OBS writes near-**linear-light** values rather than gamma-encoded sRGB. The preprocess shader corrects for this.

---

### Step 2 — Preprocess (`preprocess.hlsl`)

**Purpose:** resize the full W×H source frame into a 256×256 FP32 NCHW tensor, with correct normalization for the model.

**Constant buffer layout (b0):**
```
uint2  InputSize   = (W, H)          // full source dimensions
uint2  OutputSize  = (256, 256)      // fixed model input size
float2 Scale       = (W/256, H/256)  // per-axis, independent (no cropping)
float2 Offset      = (0, 0)
```

**Shader behavior:**
1. Each thread handles one output pixel `(x, y)`.
2. Computes `texCoord = ((x+0.5)*Scale.x/W,  (y+0.5)*Scale.y/H)` — maps all 256×256 pixels to the full input frame without letterboxing.
3. Samples `InputTexture` (BGRA UNORM) via `LinearSampler`.
4. Applies approximate linear→sRGB encoding: `rgb = pow(saturate(rgb), 1/2.2)` to compensate for OBS rendering linear-light values into the UNORM texrender.
5. Writes `asuint(r)`, `asuint(g)`, `asuint(b)` to planes 0, 1, 2 of the NCHW output buffer (RGB order).

**Dispatch:** `Dispatch(16, 16, 1)` with `numthreads(16, 16, 1)` = 256×256 total threads.

**Critical D3D11 requirement:** `OMSetRenderTargets(0, nullptr, nullptr)` **must be called before `CSSetShaderResources`**. OBS leaves `input_d3d11` (the texrender texture) bound as an RTV in the OM stage after `gs_texrender_end`. D3D11's hazard detection silently NULLs a CS SRV if the same resource is simultaneously bound as an RTV — causing `SampleLevel` to return (0,0,0,0) for every pixel, giving all-zero TRT input.

---

### Step 3 — D3D11 → CUDA interop

```cpp
d3d11_context->Flush();                          // submit preprocess Dispatch to GPU
cudaGraphicsMapResources(2, resources, stream);  // CUDA takes ownership
cudaGraphicsResourceGetMappedPointer(...);       // get device pointers
```

Both `trt_input_buffer` and `trt_output_buffer` are mapped together.

---

### Step 4 — TensorRT inference

```cpp
context->setTensorAddress(input_name,  d_input_ptr);
context->setTensorAddress(output_name, d_output_ptr);
context->enqueueV3(stream);
```

`enqueueV3` is asynchronous — it returns immediately and execution continues on the CUDA stream. The synchronize step (Step 5) waits for completion.

**Tensor I/O types:** Both input and output tensors are **`DataType::kFLOAT` (FP32)**. The "fp16" in the engine filename refers to internal layer precision, not I/O format. This was the source of a major correctness bug when FP16 packing was assumed. Verify at startup:
```
[TensorRT] Input tensor '...': ... (type: float)
[TensorRT] Output tensor '...': ... (type: float)
```

---

### Step 5 — CUDA → D3D11 interop

```cpp
cudaStreamSynchronize(stream);          // wait for inference to complete
cudaGraphicsUnmapResources(2, ...);     // return ownership to D3D11
```

After `cudaGraphicsUnmapResources`, D3D11 can safely read from `trt_output_buffer` via its SRV.

---

### Step 6 — Postprocess (`postprocess.hlsl`)

**Purpose:** resample the 1024×1024 FP32 NCHW tensor back to the OBS output resolution W×H.

**Constant buffer layout (b0):**
```
uint2  OutputSize  = (W, H)         // final output dimensions
uint2  TensorSize  = (1024, 1024)   // TRT output dimensions (fixed)
```

**Shader behavior:**
1. Each thread handles one output pixel `(x, y)`.
2. Computes `tensorUV = (x+0.5) × (TensorSize/OutputSize)` — independent per-axis scaling, so 1024×1024 maps to W×H without aspect-ratio distortion.
3. Bilinear-samples the tensor via `SampleTensorBilinear()` using `ReadFP32(n) = asfloat(InputBuffer[n])`.
4. Writes `float4(rgb.b, rgb.g, rgb.r, 1.0)` to `OutputTexture` (BGRA UNORM UAV).

**Dispatch:** `Dispatch((W+15)/16, (H+15)/16, 1)` with `numthreads(16, 16, 1)`.

---

### Step 7 — Copy and draw

```cpp
d3d11_context->CopyResource(gs_tex_d3d11, output_d3d11_texture);
```

`output_texture` (the `gs_texture_t`) and `output_d3d11_texture` are **separate objects** — `CopyResource` is required every frame. After the copy:

```cpp
obs_source_process_filter_begin(context, GS_BGRA_UNORM, OBS_ALLOW_DIRECT_RENDERING);
gs_effect_set_texture(image_param, output_texture);
while (gs_effect_loop(default_effect, technique_name))
    gs_draw_sprite(NULL, 0, W, H);
// NOTE: do NOT call obs_source_process_filter_end — it re-renders the upstream source
// over the already-drawn TRT output, overwriting it completely.
```

**No explicit D3D11 barriers are needed** between postprocess `Dispatch`, `CopyResource`, and draw. The D3D11 immediate context serializes all commands in submission order.

---

## Tensor Formats and Shapes

| Buffer | Shape | Element type | Bytes | Notes |
|--------|-------|-------------|-------|-------|
| `trt_input_buffer` | 1×3×256×256 | `float` (FP32) | 786 432 | Written by preprocess, read by TRT |
| `trt_output_buffer` | 1×3×1024×1024 | `float` (FP32) | 12 582 912 | Written by TRT, read by postprocess |

Both buffers use D3D11 structured buffers with `StructureByteStride = sizeof(uint32_t) = 4`, which equals `sizeof(float)`. The HLSL structured buffer is declared `StructuredBuffer<uint>` / `RWStructuredBuffer<uint>`; `asfloat()` / `asuint()` are used for type reinterpretation.

---

## Required Synchronization Points

| Point | Mechanism | Why required |
|-------|-----------|-------------|
| After preprocess Dispatch, before CUDA map | `d3d11_context->Flush()` | Submits D3D11 commands to GPU before CUDA takes ownership |
| After TRT enqueue, before D3D11 postprocess | `cudaStreamSynchronize()` | Waits for TRT to finish writing output before D3D11 reads it |
| Before binding preprocess CS SRV | `OMSetRenderTargets(0, nullptr, nullptr)` | Breaks D3D11 hazard that silently NULLs SRVs when resource is an active RTV |
| Before metrics CS dispatch | `OMSetRenderTargets(0, nullptr, nullptr)` | Same hazard applies to the metrics pass |

**Not required (removed during optimization):**
- D3D11 event queries between Dispatch → CopyResource → draw — the immediate context serializes these automatically.
- `Flush()` after postprocess or after CopyResource.

---

## Critical Pitfalls and Recent Fixes

| # | Bug | Root cause | Fix |
|---|-----|-----------|-----|
| 1 | OBS shows upstream source instead of TRT output | `obs_source_process_filter_end` re-renders upstream source over the manual draw | Removed the `_end` call entirely |
| 2 | Solid black output | TRT I/O is FP32, but shaders used packed-FP16 encoding | Rewrote preprocess to write `asuint(float)`, postprocess to read `asfloat(InputBuffer[n])` |
| 3 | `CreateShaderResourceView` silently failed | Input texrender format is `DXGI_FORMAT_B8G8R8A8_UNORM` (87), SRV was hardcoded to `DXGI_FORMAT_B8G8R8A8_UNORM` — format changed after `GS_CS_SRGB` context was added | Use `GetDesc()` to query actual texture format for the SRV |
| 4 | Sampling returns zeros despite valid SRV | D3D11 hazard detection NULLs CS t0 when the resource is simultaneously an active RTV | Add `OMSetRenderTargets(0, nullptr, nullptr)` before `CSSetShaderResources` |

Full details: [`PIPELINE_FIXES.md`](../PIPELINE_FIXES.md).

### Pitfall: Shared Constant Buffer

`filter->constant_buffer` is reused by both the preprocess and postprocess passes. Caching writes independently per-pass is incorrect — preprocess runs first and caches its values, then postprocess overwrites the buffer; on the next frame, preprocess sees a cache hit and skips the `Map`, but the buffer now contains postprocess values. This causes the preprocess shader to use wrong dimensions and overflow the input buffer by ~16×.

**Current behavior:** constant buffer is updated unconditionally every frame for both passes.

---

## Metrics System

The optional metrics pass (`data/metrics.hlsl`) samples a sparse `N×N` grid (default 64×64) from both the source and the TRT output and computes:

| Metric | What it measures |
|--------|----------------|
| `out_sharpness` | Mean \|Laplacian(output luma)\| — higher = sharper TRT output |
| `in_sharpness` | Mean \|Laplacian(input luma)\| — baseline before SR |
| `sharpness_delta` | `out_sharpness − in_sharpness` — **positive = SR added detail** |
| `out_mean_lum` | Mean output luminance — detects catastrophic failures (all-black: ~0, blown out: ~1) |
| `temporal_flicker` | `|out_mean_lum_t − out_mean_lum_{t−1}|` — CPU-computed; high = temporal instability |

**Note:** MAE/MSE/PSNR against the source frame are not meaningful SR quality metrics. A better SR model produces a *higher* source-vs-output MAE (more detail added). True PSNR/MAE requires a ground-truth HR reference image, which does not exist in a live pipeline.

---

## Known Limitations and Open Issues

1. **Engine path is hardcoded** (`realesrgan_256_fp16.engine`). Should be exposed as an OBS filter property.

2. **Constant buffer caching not safe** across the two passes until a second, per-pass constant buffer is introduced. Currently reverted to per-frame unconditional Map.

3. **`output_texture_uav` per-frame allocation** — creating a UAV each frame for `output_d3d11_texture` is mildly wasteful. Can be safely cached once the cbuffer issue is resolved.

4. **Linear-light input to model** — the texrender stores linear-light values (OBS renders in linear light to a UNORM texrender without `_SRGB` format). The preprocess applies `pow(x, 1/2.2)` to convert to approximate sRGB before feeding TRT. Whether this matches the model's exact training normalization is unverified.

5. **Single filter instance assumption** — `static` cache variables in the render function are shared across all filter instances in the same process.

6. **No error recovery** — if TRT inference fails mid-stream, the filter calls `obs_source_skip_video_filter` (passthrough). There is no state reset or re-initialization.
