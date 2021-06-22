#include "GPUShadingKernels.cuh"
#include <device_launch_parameters.h>

CPU_ON_GPU void MergeOutputChannels(
    const uint2 a_Resolution,
    const cudaSurfaceObject_t a_Input,
    const cudaSurfaceObject_t a_Output,
    const bool a_BlendOutput,
    const unsigned a_BlendCount
)
{
    const unsigned int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    constexpr unsigned int numChannels = static_cast<unsigned>(LightChannel::NUM_CHANNELS);

    if (pixelX < a_Resolution.x && pixelY < a_Resolution.y)
    {
        

        float4 mergedColor = { 0.f };
#pragma unroll 
        for(unsigned int channelIndex = 0; channelIndex < numChannels; ++channelIndex)
        {

            float4 channelColor{ 0.f };
            surf2DLayeredread<float4>(
                &channelColor,
                a_Input,
                pixelX * sizeof(float4),
                pixelY,
                channelIndex,
                cudaBoundaryModeTrap);

            mergedColor += channelColor;

        }

        //If enabled, average between frames.
        if(a_BlendOutput)
        {
            float4 oldValue = { 0.f };

            surf2Dread<float4>(
                &oldValue,
                a_Output,
                pixelX * sizeof(float4),
                pixelY,
                cudaBoundaryModeTrap);

            //Average results over the total blended frame count (so every frame counts just as much).
            float4 newValue = ((oldValue * static_cast<float>(a_BlendCount)) + mergedColor) / static_cast<float>(a_BlendCount + 1);
            surf2Dwrite<float4>(
                newValue,
                a_Output,
                pixelX * sizeof(float4),
                pixelY,
                cudaBoundaryModeTrap);
            
        }
        //No blending so instead overwrite previous frame data.
        else
        {

            surf2Dwrite<float4>(
                mergedColor,
                a_Output,
                pixelX * sizeof(float4),
                pixelY,
                cudaBoundaryModeTrap);

        }

    }
}