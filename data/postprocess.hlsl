// Postprocess compute shader
// Reads FP16 NCHW tensor (1024x1024), converts to BGRA8 UNORM, resamples to output texture size (W×H)
// Resample policy: Bilinear interpolation from 1024×1024 → W×H

#define DEBUG_POSTPROCESS 1  // Set to 1 to write solid magenta for debugging

StructuredBuffer<uint> InputBuffer : register(t0);  // FP16 values (one per element, stored as uint)
RWTexture2D<float4> OutputTexture : register(u0);

cbuffer Constants : register(b0)
{
    uint2 OutputSize;     // Width, Height of output texture (W×H, original OBS frame size)
    uint2 TensorSize;     // 1024, 1024 (TRT output tensor size)
};

// Bilinear sample from planar FP16 CHW tensor
float3 SampleTensorBilinear(float2 tensorUV)
{
    // Clamp to valid range
    tensorUV = clamp(tensorUV, float2(0.0, 0.0), float2(TensorSize.x - 1.0, TensorSize.y - 1.0));
    
    // Get integer coordinates and fractional parts
    float2 texelCoord = floor(tensorUV);
    float2 frac = tensorUV - texelCoord;
    
    // Clamp to valid integer coordinates
    uint2 coord00 = uint2(clamp(texelCoord, float2(0.0, 0.0), float2(TensorSize.x - 1, TensorSize.y - 1)));
    uint2 coord10 = uint2(clamp(texelCoord + float2(1.0, 0.0), float2(0.0, 0.0), float2(TensorSize.x - 1, TensorSize.y - 1)));
    uint2 coord01 = uint2(clamp(texelCoord + float2(0.0, 1.0), float2(0.0, 0.0), float2(TensorSize.x - 1, TensorSize.y - 1)));
    uint2 coord11 = uint2(clamp(texelCoord + float2(1.0, 1.0), float2(0.0, 0.0), float2(TensorSize.x - 1, TensorSize.y - 1)));
    
    // Calculate indices for NCHW layout
    uint channelSize = TensorSize.x * TensorSize.y;
    uint idx00 = coord00.y * TensorSize.x + coord00.x;
    uint idx10 = coord10.y * TensorSize.x + coord10.x;
    uint idx01 = coord01.y * TensorSize.x + coord01.x;
    uint idx11 = coord11.y * TensorSize.x + coord11.x;
    
    // Read FP16 values for all 4 corners and 3 channels
    float3 c00_rgb, c10_rgb, c01_rgb, c11_rgb;
    
    c00_rgb.r = f16tof32(InputBuffer[0 * channelSize + idx00] & 0xFFFF);
    c00_rgb.g = f16tof32(InputBuffer[1 * channelSize + idx00] & 0xFFFF);
    c00_rgb.b = f16tof32(InputBuffer[2 * channelSize + idx00] & 0xFFFF);
    
    c10_rgb.r = f16tof32(InputBuffer[0 * channelSize + idx10] & 0xFFFF);
    c10_rgb.g = f16tof32(InputBuffer[1 * channelSize + idx10] & 0xFFFF);
    c10_rgb.b = f16tof32(InputBuffer[2 * channelSize + idx10] & 0xFFFF);
    
    c01_rgb.r = f16tof32(InputBuffer[0 * channelSize + idx01] & 0xFFFF);
    c01_rgb.g = f16tof32(InputBuffer[1 * channelSize + idx01] & 0xFFFF);
    c01_rgb.b = f16tof32(InputBuffer[2 * channelSize + idx01] & 0xFFFF);
    
    c11_rgb.r = f16tof32(InputBuffer[0 * channelSize + idx11] & 0xFFFF);
    c11_rgb.g = f16tof32(InputBuffer[1 * channelSize + idx11] & 0xFFFF);
    c11_rgb.b = f16tof32(InputBuffer[2 * channelSize + idx11] & 0xFFFF);
    
    // Bilinear interpolation
    float3 c0 = lerp(c00_rgb, c10_rgb, frac.x);
    float3 c1 = lerp(c01_rgb, c11_rgb, frac.x);
    float3 result = lerp(c0, c1, frac.y);
    
    return result;
}

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= OutputSize.x || id.y >= OutputSize.y)
        return;
    
    // Calculate source coordinates in tensor (1024×1024)
    // Map output pixel (id.x, id.y) to tensor coordinates
    // Using bilinear resampling: 1024×1024 → W×H
    float2 tensorUV = (float2(id.xy) + 0.5) * (float2(TensorSize) / float2(OutputSize));
    
#if DEBUG_POSTPROCESS
    // DEBUG: Write solid magenta to prove postprocess is executing
    // Magenta in BGRA format: B=1.0, G=0.0, R=1.0, A=1.0
    OutputTexture[id.xy] = float4(1.0, 0.0, 1.0, 1.0);
#else
    // Sample with bilinear interpolation
    float3 rgb = SampleTensorBilinear(tensorUV);
    
    // Clamp to [0,1] and convert to BGRA
    rgb = saturate(rgb);

    // DEBUG: prove postprocess output is rendered
    rgb = 1.0 - rgb;
    
    // Write BGRA (swap R and B channels for D3D11 BGRA format)
    OutputTexture[id.xy] = float4(rgb.b, rgb.g, rgb.r, 1.0);
#endif
}
