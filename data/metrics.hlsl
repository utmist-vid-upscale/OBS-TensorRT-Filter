// Metrics compute shader
// Computes MAE and MSE between input and output textures
// Samples original textures at downsampled coordinates
// Per-thread-group reduction, then CPU sums the group results

Texture2D<float4> InputTexture : register(t0);  // Original input texture (W×H)
Texture2D<float4> OutputTexture : register(t1);  // Original output texture (W×H)
RWStructuredBuffer<float2> GroupSums : register(u0);  // Per-group sums: [mae_sum, mse_sum]

SamplerState LinearSampler : register(s0);

cbuffer Constants : register(b0)
{
    uint2 SampleSize;  // Sample size (64 or 128)
    uint2 InputSize;   // Original texture size (W×H)
};

groupshared float2 s_sum[256];  // Shared memory for per-group reduction (16x16 threads)

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint groupIndex : SV_GroupIndex)
{
    uint2 coord = id.xy;
    float2 pixel_val = float2(0.0, 0.0);  // [mae, mse] for this pixel
    
    // Process pixel if within bounds
    if (coord.x < SampleSize.x && coord.y < SampleSize.y) {
        // Calculate UV coordinates for original textures (downsample by sampling sparsely)
        float2 uv = (float2(coord) + 0.5) / float2(SampleSize);
        
        // Sample both textures at downsampled coordinates
        float4 in_bgra = InputTexture.SampleLevel(LinearSampler, uv, 0.0);
        float4 out_bgra = OutputTexture.SampleLevel(LinearSampler, uv, 0.0);
        
        // Convert BGRA to RGB
        float3 in_rgb = in_bgra.bgr;
        float3 out_rgb = out_bgra.bgr;
        
        // Compute per-channel absolute difference
        float3 diff = abs(in_rgb - out_rgb);
        
        // Compute per-channel squared difference
        float3 diff_sq = (in_rgb - out_rgb) * (in_rgb - out_rgb);
        
        // Average across channels (MAE and MSE per pixel)
        pixel_val.x = (diff.r + diff.g + diff.b) / 3.0;
        pixel_val.y = (diff_sq.r + diff_sq.g + diff_sq.b) / 3.0;
    }
    
    // Store in shared memory
    s_sum[groupIndex] = pixel_val;
    
    GroupMemoryBarrierWithGroupSync();
    
    // Tree reduction within thread group (256 threads = 16x16)
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (groupIndex < stride) {
            s_sum[groupIndex] += s_sum[groupIndex + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    
    // First thread in group writes group sum to buffer
    if (groupIndex == 0) {
        // Calculate group ID: for 64x64 we have 4x4 groups, for 128x128 we have 8x8 groups
        uint num_groups_x = (SampleSize.x + 15) / 16;
        uint group_id = gid.y * num_groups_x + gid.x;
        GroupSums[group_id] = s_sum[0];
    }
}
