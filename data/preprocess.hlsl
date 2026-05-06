// Preprocess compute shader
// Reads BGRA UNORM texture, resizes/crops to 256x256, converts to RGB, normalizes to [0,1], writes FP32 NCHW
//
// Layout: one float per uint32 element, NCHW order.
// numthreads(16,16,1) x Dispatch(16,16,1) = 256x256 threads, one pixel per thread.

Texture2D<float4> InputTexture : register(t0);
RWStructuredBuffer<uint> OutputBuffer : register(u0);

SamplerState LinearSampler : register(s0);

cbuffer Constants : register(b0)
{
    uint2 InputSize;      // Width, Height of input texture
    uint2 OutputSize;     // 256, 256
    float2 Scale;         // Scale factor for resizing
    float2 Offset;        // Offset for centering
};

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint x = id.x;
    uint y = id.y;

    if (x >= OutputSize.x || y >= OutputSize.y)
        return;

    uint channelSize = OutputSize.x * OutputSize.y;
    uint pixelIndex  = y * OutputSize.x + x;

    float2 uv       = (float2(x, y) + 0.5) * Scale + Offset;
    float2 texCoord = uv / float2(InputSize);
    float3 rgb      = InputTexture.SampleLevel(LinearSampler, texCoord, 0.0).bgr;

    // OBS renders into GS_CS_SRGB texrender but stores in UNORM (no _SRGB format),
    // so sampled values are linear-light. RealESRGAN expects sRGB-encoded [0,1].
    // Apply approximate linear→sRGB to match model training data.
    rgb = pow(saturate(rgb), 1.0 / 2.2);

    // NCHW: R plane, G plane, B plane — each plane is channelSize uint32 (float) elements
    OutputBuffer[0u * channelSize + pixelIndex] = asuint(rgb.r);
    OutputBuffer[1u * channelSize + pixelIndex] = asuint(rgb.g);
    OutputBuffer[2u * channelSize + pixelIndex] = asuint(rgb.b);
}
