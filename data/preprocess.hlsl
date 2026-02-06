// Preprocess compute shader
// Reads BGRA UNORM texture, resizes/crops to 256x256, converts to RGB, normalizes to [0,1], writes FP16 NCHW

Texture2D<float4> InputTexture : register(t0);
RWStructuredBuffer<uint> OutputBuffer : register(u0);  // FP16 values (one per element, stored as uint)

SamplerState LinearSampler : register(s0);

cbuffer Constants : register(b0)
{
    uint2 InputSize;      // Width, Height of input texture
    uint2 OutputSize;     // 256, 256
    float2 Scale;         // Scale factor for resizing
    float2 Offset;        // Offset for centering
};

// Convert float to half and pack into uint (2 halfs per uint)
uint PackHalf2(float2 v)
{
    uint2 h = f32tof16(v);
    return h.x | (h.y << 16);
}

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= OutputSize.x || id.y >= OutputSize.y)
        return;
    
    // Calculate source coordinates (centered crop)
    float2 uv = (float2(id.xy) + 0.5) * Scale + Offset;
    float2 texCoord = uv / float2(InputSize);
    
    // Sample input texture (BGRA)
    float4 bgra = InputTexture.SampleLevel(LinearSampler, texCoord, 0.0);
    
    // Convert BGRA to RGB and normalize to [0,1]
    float3 rgb = bgra.bgr;  // Swap B and R channels
    
    // NCHW layout: [N=0][C=0..2][H][W]
    // Buffer layout: one FP16 per element
    // For channel C at position (y, x), we write at: C * H * W + y * W + x
    uint pixelIndex = id.y * OutputSize.x + id.x;
    uint channelSize = OutputSize.x * OutputSize.y;
    
    // Convert to FP16 and write (f32tof16 returns uint with FP16 in lower 16 bits)
    // Mask to ensure only 16 bits are written (buffer stride is 2 bytes)
    OutputBuffer[0 * channelSize + pixelIndex] = f32tof16(rgb.r) & 0xFFFF;  // R channel
    OutputBuffer[1 * channelSize + pixelIndex] = f32tof16(rgb.g) & 0xFFFF;  // G channel
    OutputBuffer[2 * channelSize + pixelIndex] = f32tof16(rgb.b) & 0xFFFF;  // B channel
}
