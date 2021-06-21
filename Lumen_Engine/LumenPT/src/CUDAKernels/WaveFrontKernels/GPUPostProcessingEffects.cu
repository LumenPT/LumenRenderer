#include "GPUShadingKernels.cuh"
#include <device_launch_parameters.h>
#include "../../Shaders/CppCommon/Half4.h"
#include "../../../vendor/Include/Cuda/cuda/helpers.h"

#include <cassert>

CPU_ON_GPU void PostProcessingEffects()
{

}

CPU_ON_GPU void PrepareOptixDenoisingGPU(
    const uint2 a_RenderResolution,
    const cudaSurfaceObject_t a_PixelBufferSingleChannel,
    float3* a_IntermediaryInput,
    float3* a_IntermediaryOutput)
{
    const unsigned int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    const unsigned int pixelDataIndex = PIXEL_DATA_INDEX(pixelX, pixelY, a_RenderResolution.x);

    if (pixelX < a_RenderResolution.x && pixelY < a_RenderResolution.y)
    {
        //assert(pixelX < a_LaunchParams.m_RenderResolution.x && pixelY < a_LaunchParams.m_RenderResolution.y);

        half4Ushort4 color{ 0.f };

        surf2Dread<ushort4>(
            &color.m_Ushort4,
            a_PixelBufferSingleChannel,
            pixelX * sizeof(ushort4),
            pixelY,
            cudaBoundaryModeTrap);

        float4 colorFloat = color.m_Half4.AsFloat4();

        a_IntermediaryInput[pixelDataIndex] = make_float3(colorFloat);

        /*float4 color{ 0.f };

        surf2Dread<float4>(
            &color,
            a_PixelBufferSingleChannel,
            pixelX * sizeof(float4),
            pixelY,
            cudaBoundaryModeTrap);

        a_IntermediaryInput[pixelDataIndex] = make_float3(color.x, color.y, color.z);*/
        //a_IntermediaryOutput[pixelDataIndex] = make_float3(color.x, color.y, color.z); //TODO: for testing, remove

        //printf("%f %f %f\n", color.x, color.y, color.z);
    }
}

CPU_ON_GPU void FinishOptixDenoisingGPU(const uint2 a_RenderResolution,
    const cudaSurfaceObject_t a_PixelBufferSingleChannel,
    float3* a_IntermediaryInput,
    float3* a_IntermediaryOutput)
{
    const unsigned int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    const unsigned int pixelDataIndex = PIXEL_DATA_INDEX(pixelX, pixelY, a_RenderResolution.x);

    if (pixelX < a_RenderResolution.x && pixelY < a_RenderResolution.y)
    {
        float4 value = make_float4(a_IntermediaryOutput[pixelDataIndex], 1.0f);

        //color = { value.x, value.y, value.z, 1.f };
        half4Ushort4 color{ value };

        surf2Dwrite<ushort4>(
            color.m_Ushort4,
            a_PixelBufferSingleChannel,
            pixelX * sizeof(ushort4),
            pixelY,
            cudaBoundaryModeTrap);
    }
}