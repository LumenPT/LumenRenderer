#include "GPUShadingKernels.cuh"
#include <device_launch_parameters.h>

CPU_ON_GPU void MergeOutputChannels(
    const uint2 a_Resolution,
    const float3* const a_Input,
    float3* const a_Output)
{

    const unsigned int numPixels = a_Resolution.x * a_Resolution.y;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (int i = index; i < numPixels; i += stride)
    {
        const float3* first = &a_Input[i * static_cast<unsigned>(LightChannel::NUM_CHANNELS)];

        //Reset to 0.
        a_Output[i] = { 0.f };

        //Mix the results;
        for(int channel = 0; channel < static_cast<unsigned>(LightChannel::NUM_CHANNELS); ++channel)
        {
            a_Output[i] += first[channel];
        }
    }
}