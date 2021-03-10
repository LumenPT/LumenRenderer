#include "GPUShadingKernels.cuh"
#include <device_launch_parameters.h>

CPU_ON_GPU void MergeOutputChannels(
    const uint2 a_Resolution,
    const ResultBuffer* const a_Input,
    PixelBuffer* const a_Output)
{

    const unsigned int numPixels = a_Resolution.x * a_Resolution.y;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    const float3* firstPixel = a_Input->m_PixelBuffer->m_Pixels;

    const unsigned int numPixelsBuffer = a_Input->m_PixelBuffer->m_NumPixels;
    const unsigned int numChannelsPixelsBuffer = a_Input->m_PixelBuffer->m_ChannelsPerPixel;

    /*printf("NumPixelsBuffer: %i NumChannelsPixelsBuffer: %i FirstPixelPtr: %p PixelBufferPtr: %p \n",
        numPixelsBuffer, numChannelsPixelsBuffer, firstPixel, a_Input->m_PixelBuffer);*/

    for (int i = index; i < numPixels; i += stride)
    {

        //Convert the index into the screen dimensions.
        const int screenY = i / a_Resolution.x;
        const int screenX = i - (screenY * a_Resolution.x);

        //Mix the results;
        float3 mergedColor = a_Input->GetPixelCombined(i);
        a_Output->SetPixel(mergedColor, i, 0);

        //printf("MergedColor: %f, %f, %f \n", mergedColor.x, mergedColor.y, mergedColor.z);

    }

}