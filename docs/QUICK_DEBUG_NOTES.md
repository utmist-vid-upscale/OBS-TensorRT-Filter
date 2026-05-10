# Quick Debug Notes

| # | Bug | Symptom | Fix | File | Status |
|---|-----|---------|-----|------|--------|
| 1 | `obs_source_process_filter_end` re-renders upstream source over filter output | TRT output written correctly but OBS displays original frame | Remove `obs_source_process_filter_end` call after `gs_draw_sprite` | `src/plugin-main.cpp` | Fixed |
| 2 | Preprocess wrote packed FP16; postprocess decoded packed FP16 — engine I/O is FP32 | Solid black output; TRT input/output all zeros with real sampling | Rewrite preprocess to write `asuint(float)` per element; rewrite postprocess to read `asfloat(InputBuffer[n])` per element | `data/preprocess.hlsl`, `data/postprocess.hlsl` | Fixed |
| 3 | SRV created with hardcoded `DXGI_FORMAT_B8G8R8A8_UNORM`; texrender produces `UNORM_SRGB` | `CreateShaderResourceView` returned `E_INVALIDARG` silently; preprocess sampled zeros | Use `input_d3d11->GetDesc()` to get actual format; pass it to `srv_desc.Format` | `src/plugin-main.cpp` | Fixed |
| 4 | D3D11 hazard detection nulled CS t0 SRV because `input_d3d11` was still bound as RTV | Preprocess sampling returned zeros despite non-null SRV pointer and non-black source texture | Call `OMSetRenderTargets(0, nullptr, nullptr)` before `CSSetShaderResources` | `src/plugin-main.cpp` | Fixed |
| 5 | Preprocess crop math uses `max(scale_x, scale_y)` causing over-crop and letterbox | Output image vertically compressed | Change scale to `min(scale_x, scale_y)` or remove crop entirely | `src/plugin-main.cpp` | **Open** |
| 6 | `DEBUG_RAW_TRT 1` still active in postprocess shader | OBS output shows raw TRT float values, not real decoded image | Set `#define DEBUG_RAW_TRT 0` | `data/postprocess.hlsl` | **Open** |
