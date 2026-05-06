// Metrics compute shader — no-reference quality metrics for super-resolution
//
// Computes per-thread-group sums over a sparse SampleSize×SampleSize grid,
// then CPU accumulates across groups.
//
// Per-group output (float4 GroupSums[group_id]):
//   .x  sum |Laplacian(output_lum)|  — output sharpness accumulator
//   .y  sum |Laplacian(input_lum)|   — input sharpness accumulator
//   .z  sum output_lum               — output mean luminance (used for temporal flicker on CPU)
//   .w  sum input_lum                — input mean luminance (sanity / color-space offset)
//
// After CPU divides each sum by total_pixels:
//   out_sharpness  = sum.x / N   (higher = more high-frequency detail in SR output)
//   in_sharpness   = sum.y / N   (baseline before SR)
//   sharpness_delta = out - in   (positive = SR enhanced detail)
//   temporal_flicker computed on CPU from consecutive out_mean_lum values

Texture2D<float4> InputTexture  : register(t0);  // Upstream source frame (W×H)
Texture2D<float4> OutputTexture : register(t1);  // TRT upscaled output (W×H)
RWStructuredBuffer<float4> GroupSums : register(u0);

SamplerState LinearSampler : register(s0);

cbuffer Constants : register(b0)
{
    uint2 SampleSize;  // Grid of sample points (e.g. 64×64 or 128×128)
    uint2 InputSize;   // Full texture dimensions — used for texel-level Laplacian step
};

groupshared float4 s_sum[256];  // 16×16 threads per group

static const float3 kLuma = float3(0.2126, 0.7152, 0.0722);  // Rec709 luma weights

float Luma(float4 bgra)
{
    return dot(bgra.bgr, kLuma);
}

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint groupIndex : SV_GroupIndex)
{
    float4 pixel_val = float4(0, 0, 0, 0);

    if (id.x < SampleSize.x && id.y < SampleSize.y)
    {
        float2 uv   = (float2(id.xy) + 0.5) / float2(SampleSize);
        float2 step = 1.0 / float2(InputSize);  // one texel in the full-resolution texture

        // 5-tap discrete Laplacian for output luma: L = -4*c + l + r + u + d
        float out_c = Luma(OutputTexture.SampleLevel(LinearSampler, uv,                        0));
        float out_l = Luma(OutputTexture.SampleLevel(LinearSampler, uv + float2(-step.x,    0), 0));
        float out_r = Luma(OutputTexture.SampleLevel(LinearSampler, uv + float2(+step.x,    0), 0));
        float out_u = Luma(OutputTexture.SampleLevel(LinearSampler, uv + float2(    0, -step.y), 0));
        float out_d = Luma(OutputTexture.SampleLevel(LinearSampler, uv + float2(    0, +step.y), 0));
        float out_lap = abs(-4.0 * out_c + out_l + out_r + out_u + out_d);

        // 5-tap Laplacian for input luma
        float in_c = Luma(InputTexture.SampleLevel(LinearSampler, uv,                        0));
        float in_l = Luma(InputTexture.SampleLevel(LinearSampler, uv + float2(-step.x,    0), 0));
        float in_r = Luma(InputTexture.SampleLevel(LinearSampler, uv + float2(+step.x,    0), 0));
        float in_u = Luma(InputTexture.SampleLevel(LinearSampler, uv + float2(    0, -step.y), 0));
        float in_d = Luma(InputTexture.SampleLevel(LinearSampler, uv + float2(    0, +step.y), 0));
        float in_lap = abs(-4.0 * in_c + in_l + in_r + in_u + in_d);

        pixel_val = float4(out_lap, in_lap, out_c, in_c);
    }

    s_sum[groupIndex] = pixel_val;
    GroupMemoryBarrierWithGroupSync();

    // Tree reduction within thread group
    for (uint stride = 128; stride > 0; stride >>= 1)
    {
        if (groupIndex < stride)
            s_sum[groupIndex] += s_sum[groupIndex + stride];
        GroupMemoryBarrierWithGroupSync();
    }

    if (groupIndex == 0)
    {
        uint num_groups_x = (SampleSize.x + 15) / 16;
        uint group_id = gid.y * num_groups_x + gid.x;
        GroupSums[group_id] = s_sum[0];
    }
}
