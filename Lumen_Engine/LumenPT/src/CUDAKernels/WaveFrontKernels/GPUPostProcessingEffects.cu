#include "GPUShadingKernels.cuh"
#include <device_launch_parameters.h>
#include "../../../vendor/Include/Cuda/cuda/helpers.h"

CPU_ON_GPU void PostProcessingEffects()
{
}

CPU_ON_GPU void PrepareOptixDenoisingGPU(const WaveFront::OptixDenoiserLaunchParameters& a_LaunchParams)
{
    const unsigned int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    const unsigned int pixelDataIndex = PIXEL_DATA_INDEX(pixelX, pixelY, a_LaunchParams.m_RenderResolution.x);

    if (pixelX < a_LaunchParams.m_RenderResolution.x && pixelY < a_LaunchParams.m_RenderResolution.y)
    {
        float4 color{ 0.f };

        surf2Dread<float4>(
            &color,
            a_LaunchParams.m_PixelBufferSingleChannel,
            pixelX * sizeof(float4),
            pixelY,
            cudaBoundaryModeTrap);

        a_LaunchParams.m_IntermediaryInput[pixelDataIndex] = make_float3(color.x, color.y, color.z);

    }
}

CPU_ON_GPU void FinishOptixDenoisingGPU(const WaveFront::OptixDenoiserLaunchParameters& a_LaunchParams)
{
    const unsigned int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    const unsigned int pixelDataIndex = PIXEL_DATA_INDEX(pixelX, pixelY, a_LaunchParams.m_RenderResolution.x);

    //This literally copies 1-to-1 so it doesn't need to know about pixel indices or anything.
    //TODO: Maybe skip this step entirely and just directly output to this buffer when merging light channels? Then apply effects in this buffer?
    //TODO: It would save one copy.
    if (pixelX < a_LaunchParams.m_RenderResolution.x && pixelY < a_LaunchParams.m_RenderResolution.y)
    {
        float4 color{ 0.f };

        auto value = a_LaunchParams.m_IntermediaryOutput[pixelDataIndex];

        color = { value.x, value.y, value.z, 1.f };

        surf2Dwrite<float4>(
            color,
            a_LaunchParams.m_PixelBufferSingleChannel,
            pixelX * sizeof(float4),
            pixelY,
            cudaBoundaryModeTrap);
    }
}