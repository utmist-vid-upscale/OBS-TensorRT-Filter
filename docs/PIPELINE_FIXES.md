# Pipeline Fixes

## Overview

This document is a postmortem for the OBS TensorRT super-resolution filter pipeline (`obs-plugintemplate`). The plugin captures OBS source frames, runs them through a D3D11 preprocess compute shader, performs TensorRT inference via CUDA/D3D11 interop, and presents the upscaled result through a D3D11 postprocess compute shader and OBS filter draw APIs.

Four independent bugs were found and fixed in sequence. Each blocked progress until resolved. The debugging strategy throughout was: isolate exactly one pipeline boundary per iteration using a direct binary probe (solid magenta canary, constant-fill buffer write, CUDA device readback, CPU staging readback), confirm the fix, then move to the next boundary.

---

## Fix 1: OBS draw path overwrote the filter output

### Symptom
`DEBUG_POSTPROCESS 1` in `postprocess.hlsl` writes solid magenta to the UAV. The magenta was confirmed in a CPU staging readback. `CopyResource` succeeded. Yet OBS displayed the original upstream frame, not magenta.

### Root Cause
After `gs_draw_sprite`, the code called `obs_source_process_filter_end`. That function internally calls either `render_filter_bypass` or `render_filter_tex`, both of which re-render the **upstream source** into the current render target, completely overwriting the already-drawn TRT output. The manual `gs_draw_sprite` result was discarded every frame.

### Proof / Key Evidence
- Magenta confirmed in CPU staging readback of `output_texture` before the draw call.
- After the draw call OBS showed the original frame — not magenta — proving something after `gs_draw_sprite` was overwriting the output.
- Reading OBS source (`obs-source.c`): `render_filter_tex` does `gs_effect_set_texture(image, upstream_texture)` + `gs_draw_sprite`, and `render_filter_bypass` calls `obs_source_video_render(target)`. Both re-draw the upstream source.

### Fix Applied
Removed the `obs_source_process_filter_end` call entirely. The manual `gs_draw_sprite` is the complete and only rendering step needed.

```cpp
// REMOVED:
// obs_source_process_filter_end(filter->context, default_effect, 0, 0);

// Replaced with explanatory comment only.
```

### Why It Worked
Without `obs_source_process_filter_end`, the upstream source is never re-drawn after `gs_draw_sprite`. The TRT-processed texture remains as the final frame output.

### Notes
`obs_source_process_filter_begin` is still called (required to set up OBS render state for the draw). Only the `_end` call is omitted. This is an intentional deviation from the typical OBS filter pattern.

---

## Fix 2: TensorRT I/O is FP32, not FP16 — postprocess and preprocess tensor layout wrong

### Symptom
After Fix 1 confirmed the draw path worked, the output was solid black. `DEBUG_RAW_TRT 1` (which showed raw TRT buffer content as grayscale) was also black. CUDA readback of TRT output showed all zeros.

A constant-fill preprocess test (`asuint(0.25f)` written to all buffer elements) proved TRT inference ran and produced non-zero output, so the inference engine itself was fine.

### Root Cause
The engine filename (`realesrgan_256_fp16.engine`) implies FP16, but TensorRT runtime logs reported:

```
Input tensor type: FLOAT  (786432 bytes for 196608 elements = 4 bytes/element)
Output tensor type: FLOAT (12582912 bytes for 3145728 elements = 4 bytes/element)
```

Both I/O tensors are FP32. The "fp16" in the filename refers to internal layer precision, not I/O type.

**Preprocess** was writing packed FP16 (two `float16` values per `uint32` via `PackHalf2`). TRT was reading this as FP32 — the bit patterns for FP16 values interpreted as FP32 floats produce garbage. Result: all-zero or garbage TRT input.

**Postprocess** used `ReadFP16(n)` which unpacked two FP16 values from each `uint32` using `n >> 1` indexing. Applied to FP32 output, the lower 16 bits of each FP32 value are near-zero mantissa bits — resulting in near-zero decoded color values, i.e., a black image.

### Proof / Key Evidence
- TRT init log: `Output: 3145728 elements (12582912 bytes)` = 3145728 × 4 = 12MB. FP32.
- With `DEBUG_CONSTANT_FILL 1` (writes `asuint(0.25f)` directly, bypasses sampling): `TRT input[0..7]: 0.2500 ...` and TRT output became non-zero. This proved the pipeline was correct when given valid FP32 input.
- With real preprocess sampling (FP16 packed): `TRT input[0..7]: 0.0000 ...`.

### Fix Applied
**`preprocess.hlsl`**: Removed `PackHalf2`. Changed to one pixel per thread (`numthreads(16,16,1)`, one pixel per thread instead of 2), writing FP32 directly via `asuint(float_value)`:

```hlsl
OutputBuffer[0u * channelSize + pixelIndex] = asuint(rgb.r);
OutputBuffer[1u * channelSize + pixelIndex] = asuint(rgb.g);
OutputBuffer[2u * channelSize + pixelIndex] = asuint(rgb.b);
```

**`postprocess.hlsl`**: Replaced `ReadFP16(n)` with direct FP32 read:

```hlsl
float ReadFP32(uint n) {
    return asfloat(InputBuffer[n]);
}
```

All `SampleTensorBilinear` channel reads updated to `ReadFP32`. Buffer comment updated.

No C++ changes were needed — `sizeof(uint32_t) == sizeof(float) == 4`, so buffer allocations and UAV/SRV element counts were already correct.

### Why It Worked
TRT writes one `float` per buffer element. `asfloat(InputBuffer[n])` reinterprets the stored bits as float directly, which is what TRT wrote. The FP16 path was reinterpreting FP32 bit patterns as two packed half-floats, yielding garbage values.

---

## Fix 3: Preprocess SRV creation failed when texture format was DXGI_FORMAT_B8G8R8A8_UNORM_SRGB

### Symptom
After Fix 2, the postprocess shader received non-zero output from TRT when constant-fill was used, but real texture sampling in preprocess still produced `TRT input[0..7]: 0.0000`. The preprocess UAV write path was confirmed working (constant-fill test). The SRV, sampler, UAV, and shader pointers were all non-null at binding time (confirmed by a one-time binding probe).

### Root Cause
The SRV was created with a hardcoded format:

```cpp
srv_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;  // format 87
```

`gs_texrender_begin_with_color_space(..., GS_CS_SRGB)` creates a texture with format `DXGI_FORMAT_B8G8R8A8_UNORM_SRGB` (format 91). Creating a `DXGI_FORMAT_B8G8R8A8_UNORM` SRV on a `DXGI_FORMAT_B8G8R8A8_UNORM_SRGB` texture returns `E_INVALIDARG` and a null SRV pointer. The original code had no HRESULT check, so this failed silently. With a null SRV bound to CS t0, `SampleLevel` returned `(0,0,0,0)`.

### Proof / Key Evidence
- Added HRESULT check on `CreateShaderResourceView` — it failed with a format mismatch error.
- One-time log of `input_tex_desc.Format` showed format 91, not 87.

### Fix Applied
Query the texture's actual format at runtime and use it for the SRV:

```cpp
D3D11_TEXTURE2D_DESC input_tex_desc = {};
input_d3d11->GetDesc(&input_tex_desc);

srv_desc.Format = input_tex_desc.Format;  // use actual format, not assumed UNORM
```

Added null check for `input_d3d11` and HRESULT check on `CreateShaderResourceView` with early return on failure.

### Why It Worked
The SRV format matches the texture format, so `CreateShaderResourceView` succeeds. The shader can now sample from the correct resource.

---

## Fix 4: D3D11 hazard detection silently nulled the preprocess CS SRV

### Symptom
After Fix 3, `CreateShaderResourceView` succeeded. SRV pointer was non-null. CPU staging readback of `input_d3d11` at the center pixel showed non-zero BGRA values. Yet `TRT input[0..7]` was still `0.0000` with both `SampleLevel(0.5, 0.5)` and `Load(int3(0,0,0))` in the shader.

### Root Cause
D3D11 hazard detection. When `CSSetShaderResources(0, 1, &input_srv)` is called, D3D11 checks if the resource behind `input_srv` (`input_d3d11`) is simultaneously bound as a render target view (RTV) in the OM stage. If it is, D3D11 **silently sets CS t0 to null** to prevent a read/write hazard.

After `gs_texrender_end`, OBS restores its previous render target state. `input_d3d11` (the texrender texture) may remain bound in OBS's saved RT stack during this restoration. The `input_srv` COM object pointer remains non-null and valid — the hazard nulling happens internally to the D3D11 context slot, not to the COM object. The binding probe logging `input_srv` showed a non-null pointer, which was accurate but did not reflect what the GPU's CS t0 slot actually contained.

The `Load(int3(0,0,0))` test (which bypasses the sampler entirely) was added specifically to determine whether the SRV *slot* was null vs whether the sampler was broken. It returned zeros, confirming the SRV slot itself was null at shader time.

### Proof / Key Evidence
- `Load(int3(0,0,0))` returned zeros despite non-null SRV pointer and confirmed non-zero source texture content.
- `Load` bypasses the sampler entirely — zeros with `Load` can only mean the SRV slot is null at dispatch time.
- After adding `OMSetRenderTargets(0, nullptr, nullptr)` before `CSSetShaderResources`, `TRT input[0..7]` became non-zero on the very next frame.

### Fix Applied
Explicitly unbind all OM render targets immediately before the preprocess CS binding:

```cpp
// Break any hazard that would null the CS SRV.
// OBS leaves input_d3d11 bound as RTV after restoring its RT stack;
// obs_source_process_filter_begin (Step 7) re-establishes correct RT state.
filter->d3d11_context->OMSetRenderTargets(0, nullptr, nullptr);

filter->d3d11_context->CSSetShader(filter->preprocess_shader, nullptr, 0);
filter->d3d11_context->CSSetConstantBuffers(0, 1, &filter->constant_buffer);
filter->d3d11_context->CSSetShaderResources(0, 1, &input_srv);
```

### Why It Worked
With no RTV bound in the OM stage, D3D11's hazard checker finds no conflict and leaves CS t0 bound as set. The preprocess shader now sees the correct SRV and sampling returns real pixel values.

### Notes
`obs_source_process_filter_begin` (called later in Step 7) re-establishes OBS's render target before any OBS draw calls, so clearing the RT here does not break the output presentation path.

---

## Remaining Known Issues

1. **Vertically compressed output.** The upscaled video is visible but the image appears vertically squeezed. The preprocess crop/resize math computes a uniform scale from `max(width/256, height/256)` and applies a vertical offset of `(height - 256 * scale) / 2`, which is negative for typical 16:9 sources. This causes the shader to over-crop the top and bottom of the frame, producing a stretched or letterboxed result.

2. **`DEBUG_RAW_TRT 1` still active in `postprocess.hlsl`.** The postprocess shader is still in debug mode and is not showing the real decoded upscaled image. This must be set to `0` to enable real output.

3. **Debug probes still in code.** The CUDA input/output buffer probes (every 120 frames), the one-time binding probe log, and the source texture staging readback are still present. These should be removed or gated behind `TRT_VERBOSE 0` once the pipeline is confirmed working end-to-end.

---

## Recommended Next Debug Steps

1. **Disable `DEBUG_RAW_TRT`** in `postprocess.hlsl` (`set to 0`) to enable real decoded output.

2. **Fix preprocess crop math.** The current formula uses `scale = max(width/256, height/256)` which crops both dimensions to fit. For a 1920×1080 source this produces `scale = 7.5`, `offset_y = -420`, causing ~55 rows of letterboxing top and bottom (sampled at the clamped edge). The correct approach for most SR models is to use `scale = min(width/256, height/256)` (letterbox-scale, no cropping) or to pass the actual texture dimensions to the model and use the full frame.

3. **Remove debug probes** once output is confirmed correct. The CUDA readback probe, staging readback, and binding probe add per-frame overhead (especially the `cudaMemcpy` DeviceToHost). Gate them behind `TRT_VERBOSE 0` or remove outright.

4. **Verify model input range.** RealESRGAN typically expects pixel values normalized to `[0, 1]`. The current preprocess writes `rgb` values sampled as `float4.bgr` with no additional normalization — this is correct if OBS textures contain UNORM values (which they do). Confirm by checking TRT input values in the probe: they should be in `[0, 1]`.

5. **Validate postprocess UV mapping end-to-end.** After fixing the crop math, verify the bilinear resample in `postprocess.hlsl` maps the 1024×1024 TRT output back to the OBS source dimensions without distortion. The current formula `tensorUV = (id.xy + 0.5) * (TensorSize / OutputSize)` is correct in principle but should be validated visually once the input crop is fixed.
