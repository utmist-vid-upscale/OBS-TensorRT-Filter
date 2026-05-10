# CLAUDE.md

## Project Context
- OBS Studio video filter plugin that runs TensorRT super-resolution on GPU frames via D3D11 compute preprocess/postprocess + CUDA/TensorRT inference, then renders the processed texture back through OBS filter draw APIs.

## How to Work on This Project
- Read the minimum necessary files; prefer targeted reads over broad exploration.
- For debugging tasks, start with `DEBUG_FOCUS.md`.
- Expand to `HANDOFF.md` only when `DEBUG_FOCUS.md` lacks needed context.
- Do not scan the full repo unless explicitly required by the task.
- Prioritize critical paths first: `src/plugin-main.cpp`, `src/trt_runner.cpp`, `data/postprocess.hlsl`, `data/preprocess.hlsl`.

## Build & Run
- Environment (Windows): Visual Studio 2022, CMake >= 3.28, NVIDIA CUDA Toolkit (project preset points to CUDA v12.1), TensorRT 10.x, OBS Studio (D3D11 backend).
- Configure:
  - `cmake --preset windows-x64`
- Build:
  - `cmake --build --preset windows-x64`
- Install plugin into OBS path (run from elevated shell):
  - `cmake --install build_x64 --config RelWithDebInfo --prefix "%ProgramData%\obs-studio\plugins"`
- Engine requirement:
  - Ensure `realesrgan_256_fp16.engine` is present in the plugin module data path (`obs_module_file` lookup).
- Runtime validation:
  - Add filter in OBS and inspect logs for `[TRT Filter]` / `[TensorRT]`.

## Debugging Workflow
- Always follow this order:
  1. Identify execution path and exact failing boundary.
  2. Form 2-3 concrete hypotheses.
  3. Inspect the minimal set of files/resources needed to test hypotheses.
  4. Propose the fix before editing code.
- Prefer canary checks and resource identity verification at boundaries (dispatch -> copy -> draw).
- Confirm assumptions with logs/readbacks before making structural changes.
- Keep one variable change per iteration when isolating rendering/inference bugs.

## Critical Architecture Notes
- Inference happens in `src/trt_runner.cpp` (`setTensorAddress` + `enqueueV3`) and is orchestrated from `trt_filter_render` in `src/plugin-main.cpp`.
- Rendering/output handoff happens in `trt_filter_render`: postprocess writes UAV (`output_d3d11_texture`), then `CopyResource` into OBS `output_texture`, then OBS effect draw (`obs_source_process_filter_begin/end`).
- Data flow constraints:
  - D3D11 preprocess -> CUDA map -> TensorRT enqueue -> CUDA sync/unmap -> D3D11 postprocess -> OBS draw.
  - D3D11/CUDA interop sync correctness is required; stale or unsynchronized resources can appear as "inference succeeded but frame unchanged".
  - Shader tensor layout/precision assumptions must match TensorRT output dtype/layout.

## Known Pitfalls
- TensorRT output dtype/layout mismatch vs shader decode assumptions (FP16/FP32, CHW packing) causes garbage/flat output.
- Buffer stride/byte-size mismatches between D3D11 structured buffers and TensorRT tensor sizes silently corrupt output.
- Resource ownership ambiguity between explicit D3D11 output texture and OBS `gs_texture_t` backing texture can hide correct writes.
- Missing/failed `obs_source_process_filter_begin` or zero `gs_effect_loop` iterations can drop final rendering despite valid GPU data.
- CUDA/TensorRT/driver version mismatches or unsupported GPU state can fail interop/inference at runtime.
- Engine file missing or wrong filename/path prevents inference initialization.

## Code Style / Constraints
- Keep fixes surgical; avoid large refactors during active debug unless proven necessary.
- Preserve existing logging style (`[TRT Filter]`, `[TensorRT]`) and add high-signal logs only at boundaries.
- Keep shader and host-side tensor contracts explicitly aligned when changing dimensions/types/layout.
- Do not introduce broad file scans or unrelated cleanup in focused debug tasks.
- Validate behavior after each change with OBS runtime logs, not just successful compilation.

## Session Rules
- Before editing, restate:
  - current failing boundary
  - top 2-3 hypotheses
  - minimal files to inspect
- Prefer reading specific functions over whole files when possible.
- Do not make edits until the likely root cause is explained.
- After each edit, state exactly how to validate the change in OBS.
- Keep logs high-signal and boundary-focused.