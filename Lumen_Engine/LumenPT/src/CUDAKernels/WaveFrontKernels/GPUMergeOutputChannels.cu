#include "GPUShadingKernels.cuh"
#include <device_launch_parameters.h>

CPU_ON_GPU void MergeOutputChannels(
    const uint2 a_Resolution,
    const float3* const a_Input,
    float3* const a_Output,
    const bool a_BlendOutput,
    const unsigned a_BlendCount
)
{
    const unsigned int numPixels = a_Resolution.x * a_Resolution.y;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (int i = index; i < numPixels; i += stride)
    {
        const float3* first = &a_Input[i * static_cast<unsigned>(LightChannel::NUM_CHANNELS)];

        //If enabled, average between frames.
        if(a_BlendOutput)
        {
            float3 oldValue = a_Output[i];
            float3 newValue = { 0.f };
            for (int channel = 0; channel < static_cast<unsigned>(LightChannel::NUM_CHANNELS); ++channel)
            {
                newValue += first[channel];
            }
            //Average results over the total blended frame count (so every frame counts just as much).
            a_Output[i] = ((oldValue * static_cast<float>(a_BlendCount)) + newValue) / static_cast<float>(a_BlendCount + 1);
        }
        //No blending so instead overwrite previous frame data.
        else
        {
            //Reset to 0. NOT NEEDED: Already done in wavefront when blending is disabled.
            //a_Output[i] = { 0.f };

            //Mix the results.
            for(int channel = 0; channel < static_cast<unsigned>(LightChannel::NUM_CHANNELS); ++channel)
            {
                a_Output[i] += first[channel];
            }
        }

    }
}