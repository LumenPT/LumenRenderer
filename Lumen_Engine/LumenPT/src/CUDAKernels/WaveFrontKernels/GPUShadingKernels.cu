#include "GPUShadingKernels.cuh"
#include <device_launch_parameters.h>
#include "../../Shaders/CppCommon/Half4.h"
#include "../../../vendor/Include/Cuda/cuda/helpers.h"

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
        output[static_cast<unsigned>(LightChannel::DIRECT)] = make_float3(a_CurrentSurfaceData[i].m_MaterialData.m_Color);
    }
}


CPU_ON_GPU void WriteToOutput(
    const uint2 a_Resolution,
    const cudaSurfaceObject_t a_Input,
    uchar4* a_Output
)
{
    const unsigned int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    const unsigned int pixelDataIndex = PIXEL_DATA_INDEX(pixelX, pixelY, a_Resolution.x);

    //This literally copies 1-to-1 so it doesn't need to know about pixel indices or anything.
    //TODO: Maybe skip this step entirely and just directly output to this buffer when merging light channels? Then apply effects in this buffer?
    //TODO: It would save one copy.
    if(pixelX < a_Resolution.x && pixelY < a_Resolution.y)
    {
        half4Ushort4 color{ 0.f };

        surf2Dread<ushort4>(
            &color.m_Ushort4,
            a_Input,
            pixelX * sizeof(ushort4),
            pixelY,
            cudaBoundaryModeTrap);

        a_Output[pixelDataIndex] = make_color(color.m_Half4.AsFloat4());
            
    }
}