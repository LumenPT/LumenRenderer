#include "MotionVectors.cuh"
#include "../Shaders/CppCommon/WaveFrontDataStructs.h"
#include "../Shaders/CppCommon/Half2.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

CPU_ON_GPU void GenerateMotionVector(
    cudaSurfaceObject_t a_MotionVectorBuffer,
    const WaveFront::SurfaceData* a_CurrentSurfaceData,
    uint2 a_Resolution,
    sutil::Matrix4x4* a_PrevViewProjMatrix
)
{

    const unsigned int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    const unsigned int pixelDataIndex = PIXEL_DATA_INDEX(pixelX, pixelY, a_Resolution.x);

    if(pixelX < a_Resolution.x && pixelY < a_Resolution.y)
    {

        half2Ushort2 motionVector{ __float22half2_rn({0.f, 0.f}) };
        const WaveFront::SurfaceData& surfaceData = a_CurrentSurfaceData[pixelDataIndex];

        if(surfaceData.m_IntersectionT > 0.f)
        {

            const float4 currWorldPos = make_float4(surfaceData.m_Position, 1.0f);
            float2 currScreenPos{ static_cast<float>(pixelX), static_cast<float>(pixelY) };
            currScreenPos += make_float2(0.5f);
            currScreenPos.x /= static_cast<float>(a_Resolution.x);
            currScreenPos.y /= static_cast<float>(a_Resolution.y);

            float4 prevClipCoordinates = (*a_PrevViewProjMatrix) * currWorldPos;
            float3 prevNdc = make_float3(prevClipCoordinates.x, prevClipCoordinates.y, prevClipCoordinates.z) / prevClipCoordinates.w;
            float2 prevScreenPos = make_float2(
                (prevNdc.x * 0.5f + 0.5f)/* * static_cast<float>(a_Resolution.x)*/,
                (prevNdc.y * 0.5f + 0.5f)/* * static_cast<float>(a_Resolution.y)*/
            );

            motionVector.m_Half2 = __float22half2_rn((prevScreenPos - currScreenPos));
            
        }

        surf2Dwrite<ushort2>(
            motionVector.m_Ushort2,
            a_MotionVectorBuffer,
            pixelX * sizeof(ushort2),
            pixelY,
            cudaBoundaryModeTrap);

    }
}