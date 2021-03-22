#include "GPUShadingKernels.cuh"
#include <device_launch_parameters.h>

//Just temporary CUDA kernels.

CPU_ON_GPU void DEBUGShadePrimIntersections(
    const uint3 a_ResolutionAndDepth,
    const SurfaceData* a_CurrentSurfaceData,
    float3* const a_Output
)
{

    const unsigned int numPixels = a_ResolutionAndDepth.x * a_ResolutionAndDepth.y;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < numPixels; i += stride)
    {
        //Copy the diffuse color over for now for unshaded.
        auto* output = &a_Output[i * static_cast<unsigned>(LightChannel::NUM_CHANNELS)];
        output[static_cast<unsigned>(LightChannel::DIRECT)] = a_CurrentSurfaceData[i].m_Color;
    }
}


CPU_ON_GPU void WriteToOutput(
    const uint2 a_Resolution,
    const float3 * const a_Input,
    uchar4* a_Output)
{
    const unsigned int numPixels = a_Resolution.x * a_Resolution.y;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    //This literally copies 1-to-1 so it doesn't need to know about pixel indices or anything.
    //TODO: Maybe skip this step entirely and just directly output to this buffer when merging light channels? Then apply effects in this buffer?
    //TODO: It would save one copy.
    for (unsigned int i = index; i < numPixels; i += stride)
    {
        const auto color = make_color(a_Input[i]);
        a_Output[i] = color;
    }
}