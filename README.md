# OBS TensorRT Super-Resolution Filter

An OBS Studio video filter plugin that performs real-time GPU super-resolution using NVIDIA TensorRT. Each captured frame is preprocessed with a D3D11 compute shader, upscaled via TensorRT inference on a CUDA stream, and postprocessed back into the OBS rendering pipeline.

**Current status:** Upscale is visually working end-to-end. Output geometry and brightness are correct. See [Known Issues](#known-issues--next-steps) for remaining work.

---

## Pipeline Summary

| Step | What happens | Key format |
|------|-------------|------------|
| 1. Source capture | Upstream OBS source rendered into `gs_texrender_t` | BGRA UNORM (`GS_CS_SRGB` context) |
| 2. Preprocess | D3D11 compute → resize full frame to 256×256, sRGB-encode, write NCHW | FP32, structured buffer |
| 3. D3D11→CUDA | Flush + `cudaGraphicsMapResources` | Shared D3D11 structured buffer |
| 4. TRT inference | `enqueueV3` on CUDA stream | FP32 NCHW 1×3×256×256 → 1×3×1024×1024 |
| 5. CUDA→D3D11 | `cudaStreamSynchronize` + unmap | Returns ownership to D3D11 |
| 6. Postprocess | D3D11 compute → bilinear resample 1024×1024 → W×H, write BGRA | FP32 read → BGRA UNORM UAV |
| 7. Draw | `CopyResource` + OBS effect draw | BGRA UNORM |

> **Important:** TensorRT I/O tensors are **FP32**, not FP16. The engine filename contains "fp16" to indicate internal layer precision only.

---

## Quick Build & Install

**Requirements:** Windows, Visual Studio 2022, CMake ≥ 3.28, CUDA Toolkit (v12.1 in preset), TensorRT 10.x, OBS Studio (D3D11 backend).

```bash
# Configure
cmake --preset windows-x64

# Build
cmake --build --preset windows-x64

# Install to OBS (elevated shell)
cmake --install build_x64 --config RelWithDebInfo --prefix "%ProgramData%\obs-studio\plugins"
```

**Engine file:** Place `realesrgan_256_fp16.engine` in the OBS plugin module data path (resolved via `obs_module_file()`). The filter will fail to initialize without it.

**Validate:** Add the filter in OBS and check the OBS log for `[TRT Filter]` and `[TensorRT]` lines.

---

## Key Files

| File | Role |
|------|------|
| `src/plugin-main.cpp` | Filter lifecycle, per-frame pipeline, all D3D11/CUDA orchestration |
| `src/trt_runner.cpp/.h` | TensorRT engine load, `enqueueV3` wrapper |
| `data/preprocess.hlsl` | BGRA → FP32 NCHW compute shader (256×256 output) |
| `data/postprocess.hlsl` | FP32 NCHW → BGRA UAV compute shader (bilinear resample to W×H) |
| `data/metrics.hlsl` | Optional no-reference quality metrics (Laplacian sharpness, luminance) |
| `CLAUDE.md` | Session rules and debugging workflow for AI-assisted development |
| `docs/PIPELINE_OVERVIEW.md` | Full technical pipeline reference |
| `PIPELINE_FIXES.md` | Postmortem of the four major bugs fixed during development |
| `QUICK_DEBUG_NOTES.md` | One-line bug/fix/status table for quick reference |

---

## Known Issues / Next Steps

1. **Constant buffer caching is unsafe across passes** — preprocess and postprocess share `filter->constant_buffer`; caching both independently causes the preprocess shader to read postprocess values on frame 2+. Currently reverted to unconditional Map per frame.

2. **Per-frame `CreateUnorderedAccessView`** for the postprocess output — caching this as a persistent UAV was attempted but reverted due to the cbuffer regression. Safe to re-add independently.

3. **`static` cache variables** in the render function are shared across filter instances in the same process — affects multi-instance setups.

4. **Engine path is hardcoded** (`realesrgan_256_fp16.engine`) — should be exposed as an OBS filter property.

5. **TRT output scale** — verify the model's expected input normalization and output range match what the preprocess/postprocess assume.

6. **Metrics comparison is source vs. output**, not output vs. ground-truth HR reference. MAE/MSE/PSNR against source are not meaningful SR quality metrics. See `docs/PIPELINE_OVERVIEW.md` for the current metrics set.

---

## Further Reading

- [`docs/PIPELINE_OVERVIEW.md`](docs/PIPELINE_OVERVIEW.md) — full technical pipeline, tensor formats, sync requirements, pitfalls
- [`PIPELINE_FIXES.md`](PIPELINE_FIXES.md) — postmortem for the four major bugs fixed
- [`QUICK_DEBUG_NOTES.md`](QUICK_DEBUG_NOTES.md) — quick bug/fix reference table
- [`CLAUDE.md`](CLAUDE.md) — debugging workflow and session conventions
