// Postprocess compute shader
// Reads FP32 NCHW tensor (1024x1024), converts to BGRA8 UNORM, resamples to output texture size (W×H)
// Resample policy: Bilinear interpolation from 1024×1024 → W×H

#define DEBUG_POSTPROCESS 0  // Set to 1 to write solid magenta for debugging
#define DEBUG_RAW_TRT 0      // Set to 1 to show raw TRT float values as grayscale

// TRT writes packed FP32: one float per uint32 element, NCHW layout.
// Buffer element n = float at NCHW flat index n.
StructuredBuffer<uint> InputBuffer : register(t0);
RWTexture2D<float4> OutputTexture : register(u0);

cbuffer Constants : register(b0)
{
    uint2 OutputSize;     // Width, Height of output texture (W×H, original OBS frame size)
    uint2 TensorSize;     // 1024, 1024 (TRT output tensor size)
};

// Read one FP32 value from the buffer at NCHW flat index n.
float ReadFP32(uint n)
{
    return asfloat(InputBuffer[n]);
}

// Bilinear sample from planar FP32 CHW tensor
float3 SampleTensorBilinear(float2 tensorUV)
{
    tensorUV = clamp(tensorUV, float2(0.0, 0.0), float2(TensorSize.x - 1.0, TensorSize.y - 1.0));

    float2 texelCoord = floor(tensorUV);
    float2 frac = tensorUV - texelCoord;

    uint2 coord00 = uint2(clamp(texelCoord,                    float2(0,0), float2(TensorSize.x-1, TensorSize.y-1)));
    uint2 coord10 = uint2(clamp(texelCoord + float2(1.0, 0.0), float2(0,0), float2(TensorSize.x-1, TensorSize.y-1)));
    uint2 coord01 = uint2(clamp(texelCoord + float2(0.0, 1.0), float2(0,0), float2(TensorSize.x-1, TensorSize.y-1)));
    uint2 coord11 = uint2(clamp(texelCoord + float2(1.0, 1.0), float2(0,0), float2(TensorSize.x-1, TensorSize.y-1)));

    uint channelSize = TensorSize.x * TensorSize.y;
    uint idx00 = coord00.y * TensorSize.x + coord00.x;
    uint idx10 = coord10.y * TensorSize.x + coord10.x;
    uint idx01 = coord01.y * TensorSize.x + coord01.x;
    uint idx11 = coord11.y * TensorSize.x + coord11.x;

    float3 c00_rgb, c10_rgb, c01_rgb, c11_rgb;

    c00_rgb.r = ReadFP32(0u * channelSize + idx00);
    c00_rgb.g = ReadFP32(1u * channelSize + idx00);
    c00_rgb.b = ReadFP32(2u * channelSize + idx00);

    c10_rgb.r = ReadFP32(0u * channelSize + idx10);
    c10_rgb.g = ReadFP32(1u * channelSize + idx10);
    c10_rgb.b = ReadFP32(2u * channelSize + idx10);

    c01_rgb.r = ReadFP32(0u * channelSize + idx01);
    c01_rgb.g = ReadFP32(1u * channelSize + idx01);
    c01_rgb.b = ReadFP32(2u * channelSize + idx01);

    c11_rgb.r = ReadFP32(0u * channelSize + idx11);
    c11_rgb.g = ReadFP32(1u * channelSize + idx11);
    c11_rgb.b = ReadFP32(2u * channelSize + idx11);

    float3 c0 = lerp(c00_rgb, c10_rgb, frac.x);
    float3 c1 = lerp(c01_rgb, c11_rgb, frac.x);
    return lerp(c0, c1, frac.y);
}

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= OutputSize.x || id.y >= OutputSize.y)
        return;

    float2 tensorUV = (float2(id.xy) + 0.5) * (float2(TensorSize) / float2(OutputSize));

#if DEBUG_POSTPROCESS
    OutputTexture[id.xy] = float4(1.0, 0.0, 1.0, 1.0);  // solid magenta
#elif DEBUG_RAW_TRT
    // Show first float value amplified — non-zero means TRT wrote FP32 data correctly.
    float v0 = asfloat(InputBuffer[0]);
    float v1 = asfloat(InputBuffer[1]);
    OutputTexture[id.xy] = float4(saturate(v0 * 5), saturate(v1 * 5), 0, 1.0);
#else
    float3 rgb = SampleTensorBilinear(tensorUV);
    rgb = saturate(rgb);
    // Write BGRA (swap R and B for D3D11 BGRA format)
    OutputTexture[id.xy] = float4(rgb.b, rgb.g, rgb.r, 1.0);
#endif
}
