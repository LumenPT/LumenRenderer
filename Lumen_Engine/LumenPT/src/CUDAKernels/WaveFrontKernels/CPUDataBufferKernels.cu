#include "CPUDataBufferKernels.cuh"
#include "GPUDataBufferKernels.cuh"



CPU_ONLY void ResetPixelBuffer(
    PixelBuffer* a_PixelBufferDevPtr,
    unsigned a_NumPixels,
    unsigned a_ChannelsPerPixel)
{

    ResetPixelBufferMembers<<<1, 1>>>(a_PixelBufferDevPtr, a_NumPixels, a_ChannelsPerPixel);

    const int totalPixels = a_NumPixels * a_ChannelsPerPixel;
    const int blockSize = 256;
    const int numBlocks = (totalPixels + blockSize - 1) / blockSize;

    ResetPixelBufferData<<<numBlocks, blockSize>>>(a_PixelBufferDevPtr);

}

void ResetLightChannels(float3* a_Buffer, unsigned a_NumPixels, unsigned a_ChannelsPerPixel)
{
    const int blockSize = 256;
    const int numBlocks = (a_NumPixels + blockSize - 1) / blockSize;

    ResetLightChannelData<<<numBlocks, blockSize>>>(a_Buffer, a_ChannelsPerPixel, a_NumPixels);
}
