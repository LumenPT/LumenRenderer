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
    const SurfaceData* a_CurrentSurfaceData,
    const cudaSurfaceObject_t a_PixelBufferSingleChannel,
    float3* a_IntermediaryInput,
    float3* a_AlbedoInput,
    float3* a_NormalInput,
    float2* a_FlowInput,
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

        float4 albedo = a_CurrentSurfaceData[pixelDataIndex].m_MaterialData.m_Color;
        a_AlbedoInput[pixelDataIndex] = make_float3(albedo.x, albedo.y, albedo.z);
        a_NormalInput[pixelDataIndex] = a_CurrentSurfaceData[pixelDataIndex].m_Normal;
        a_FlowInput[pixelDataIndex] = make_float2(0.f, 0.f);
    }
}

CPU_ON_GPU void FinishOptixDenoisingGPU(const uint2 a_RenderResolution,
    const cudaSurfaceObject_t a_PixelBufferSingleChannel,
    float3* a_IntermediaryInput,
    float3* a_IntermediaryOutput,
    float3* a_BlendOutput,
    bool a_UseBlendOutput,
    unsigned int a_BlendCount)
{
    const unsigned int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    const unsigned int pixelDataIndex = PIXEL_DATA_INDEX(pixelX, pixelY, a_RenderResolution.x);

    if (pixelX < a_RenderResolution.x && pixelY < a_RenderResolution.y)
    {
        float4 denoisedValue = make_float4(a_IntermediaryOutput[pixelDataIndex], 1.0f);

        half4Ushort4 denoisedColor{ denoisedValue };

        if(a_UseBlendOutput)
        {
            float4 oldValue = make_float4(a_BlendOutput[pixelDataIndex], 1.0f);

            float4 newValue = ((oldValue * static_cast<float>(a_BlendCount)) + denoisedValue) / static_cast<float>(a_BlendCount + 1);
            half4Ushort4 newColor{ newValue };

            a_BlendOutput[pixelDataIndex] = make_float3(newValue);

            surf2Dwrite<ushort4>(
                newColor.m_Ushort4,
                a_PixelBufferSingleChannel,
                pixelX * sizeof(ushort4),
                pixelY,
                cudaBoundaryModeTrap);
        }
        else
        {
            a_BlendOutput[pixelDataIndex] = a_IntermediaryOutput[pixelDataIndex];

            surf2Dwrite<ushort4>(
                denoisedColor.m_Ushort4,
                a_PixelBufferSingleChannel,
                pixelX * sizeof(ushort4),
                pixelY,
                cudaBoundaryModeTrap);
        }
    }
}