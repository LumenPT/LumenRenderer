#include "GPUDataBufferKernels.cuh"
#include <device_launch_parameters.h>

CPU_ON_GPU void ResetPixelBufferMembers(
    PixelBuffer* const a_PixelBuffer,
    unsigned a_NumPixels,
    unsigned a_ChannelsPerPixel)
{

    *const_cast<unsigned*>(&a_PixelBuffer->m_NumPixels) = a_NumPixels;
    *const_cast<unsigned*>(&a_PixelBuffer->m_ChannelsPerPixel) = a_ChannelsPerPixel;

}

CPU_ON_GPU void ResetPixelBufferData(PixelBuffer* const a_PixelBuffer)
{

    const unsigned int bufferSize = a_PixelBuffer->GetSize();
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < bufferSize; i += stride)
    {

        a_PixelBuffer->m_Pixels[i] = { 0.f, 0.f, 0.f };
    }

}

CPU_ON_GPU void ResetLightChannelData(float3* a_LightData, unsigned a_NumChannels, unsigned a_NumPixels)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    //Loop over every pixel index, and then set each channel to 0,0,0.
    for (unsigned int i = index; i < a_NumPixels; i += stride)
    {
        for(int channel = 0; channel < a_NumChannels; ++channel)
        {
            a_LightData[a_NumChannels * i + channel] = float3{ 0.f, 0.f, 0.f };
        }
    }
}
