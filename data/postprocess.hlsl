// Postprocess compute shader
// Reads FP16 NCHW tensor, converts to BGRA8 UNORM, writes to output texture

StructuredBuffer<uint> InputBuffer : register(t0);  // FP16 values (one per element, stored as uint)
RWTexture2D<float4> OutputTexture : register(u0);

cbuffer Constants : register(b0)
{
    uint2 OutputSize;     // Width, Height of output texture
    uint2 TensorSize;     // 256, 256
};


[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= OutputSize.x || id.y >= OutputSize.y)
        return;
    
    // Calculate source coordinates in tensor (centered crop, may need to scale)
    float2 tensorUV = float2(id.xy) * (float2(TensorSize) / float2(OutputSize));
    uint2 tensorCoord = min(uint2(tensorUV), TensorSize - 1);
    
    // NCHW layout: [N=0][C=0..2][H][W]
    // Buffer layout: one FP16 per element
    uint tensorIndex = tensorCoord.y * TensorSize.x + tensorCoord.x;
    uint channelSize = TensorSize.x * TensorSize.y;
    
    // Read channels (buffer contains uint with FP16 in lower 16 bits)
    float r = f16tof32(InputBuffer[0 * channelSize + tensorIndex] & 0xFFFF);  // R channel
    float g = f16tof32(InputBuffer[1 * channelSize + tensorIndex] & 0xFFFF);  // G channel
    float b = f16tof32(InputBuffer[2 * channelSize + tensorIndex] & 0xFFFF);  // B channel
    
    // Clamp to [0,1] and convert to BGRA
    r = saturate(r);
    g = saturate(g);
    b = saturate(b);
    
    // Write BGRA (swap R and B)
    OutputTexture[id.xy] = float4(b, g, r, 1.0);
}
