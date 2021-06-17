#include "GPUShadingKernels.cuh"
#include "../../Shaders/CppCommon/Half4.h"
#include <device_launch_parameters.h>

CPU_ON_GPU void MergeOutputChannels(
    const uint2 a_Resolution,
    ArrayParameter<cudaSurfaceObject_t, static_cast<unsigned>(LightChannel::NUM_CHANNELS)> a_Input,
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
        

        half4Ushort4 mergedColor = { 0.f };
#pragma unroll 
        for(unsigned int channelIndex = 0; channelIndex < numChannels; ++channelIndex)
        {

            half4Ushort4 channelColor{ 0.f };

            surf2Dread<ushort4>(
                &channelColor.m_Ushort4,
                a_Input[channelIndex],
                pixelX * sizeof(ushort4),
                pixelY,
                cudaBoundaryModeTrap);

            mergedColor.m_Half4 += channelColor.m_Half4;

        }

        //If enabled, average between frames.
        if(a_BlendOutput)
        {
            half4Ushort4 oldValue = { 0.f };

            surf2Dread<ushort4>(
                &oldValue.m_Ushort4,
                a_Output,
                pixelX * sizeof(ushort4),
                pixelY,
                cudaBoundaryModeTrap);

            //Average results over the total blended frame count (so every frame counts just as much).
            half4Ushort4 newValue{ ((oldValue.m_Half4 * static_cast<float>(a_BlendCount)) + mergedColor.m_Half4) / static_cast<float>(a_BlendCount + 1) };
            surf2Dwrite<ushort4>(
                newValue.m_Ushort4,
                a_Output,
                pixelX * sizeof(ushort4),
                pixelY,
                cudaBoundaryModeTrap);
            
        }
        //No blending so instead overwrite previous frame data.
        else
        {

            surf2Dwrite<ushort4>(
                mergedColor.m_Ushort4,
                a_Output,
                pixelX * sizeof(ushort4),
                pixelY,
                cudaBoundaryModeTrap);

        }

    }
}