#include "SnapShotProcessing.cuh"
#include "../Shaders/CppCommon/WaveFrontDataStructs.h"

#include <device_launch_parameters.h>

CPU_ON_GPU void SeparateIntersectionRayBuffer(WaveFront::AtomicBuffer<WaveFront::IntersectionRayData>* a_IntersectionBuffer,
                                              uint2 a_Resolution,
                                              float3* a_OriginBuffer, float3* a_DirectionBuffer, float3* a_ContributionBuffer)
{
    const uint32_t bufferSize = a_IntersectionBuffer->GetSize();
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = blockDim.x * gridDim.x;

    for (uint32_t i = index; i < bufferSize - 1; i += stride)
    {
        WaveFront::IntersectionRayData* intersectionData = &a_IntersectionBuffer->data[index];

        const WaveFront::PixelIndex& pixelIndex = intersectionData->m_PixelIndex;
        const unsigned int pixelDataIndex = PIXEL_DATA_INDEX(pixelIndex.m_X, pixelIndex.m_Y, a_Resolution.x);

        a_OriginBuffer[pixelDataIndex] = intersectionData->m_Origin;
        a_DirectionBuffer[pixelDataIndex] = intersectionData->m_Direction;
        a_ContributionBuffer[pixelDataIndex] = intersectionData->m_Contribution;
    }
}

//CPU_ON_GPU void SeparateMotionVectorBuffer(uint64_t a_BufferSize, WaveFront::MotionVectorBuffer* a_MotionVectorBuffer,
//    float3* a_MotionVectorDirectionBuffer, float3* a_MotionVectorMagnitudeBuffer)
//{
//    const uint32_t bufferSize = a_BufferSize;
//    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
//    const uint32_t stride = blockDim.x * gridDim.x;
//
//    for (uint32_t i = index; i < bufferSize - 1; i += stride)
//    {
//        const WaveFront::MotionVectorData motionVectorData = a_MotionVectorBuffer->m_MotionVectorBuffer[i];
//
//        float2 velocity = motionVectorData.m_Velocity * 0.5f + 0.5f;
//    	
//        a_MotionVectorDirectionBuffer[i] = make_float3(velocity, 0.f);
//        a_MotionVectorMagnitudeBuffer[i] = make_float3(length(motionVectorData.m_Velocity));
//    }
//}
//
//CPU_ONLY void SeparateIntersectionRayBufferCPU(uint64_t a_BufferSize, WaveFront::AtomicBuffer<WaveFront::IntersectionRayData>* a_IntersectionBuffer,
//    uint2 a_Resolution,
//    float3* a_OriginBuffer, float3* a_DirectionBuffer, float3* a_ContributionBuffer)
//{
//    const int blockSize = 256;
//    const int numBlocks = (a_BufferSize + blockSize - 1) / blockSize;
//    SeparateIntersectionRayBuffer<<<numBlocks, blockSize>>>(a_IntersectionBuffer, a_Resolution, a_OriginBuffer, a_DirectionBuffer, a_ContributionBuffer);
//
//	cudaDeviceSynchronize();
//};
//
//CPU_ONLY void SeparateMotionVectorBufferCPU(uint64_t a_BufferSize, WaveFront::MotionVectorBuffer* a_MotionVectorBuffer,
//    float3* a_MotionVectorDirectionBuffer, float3* a_MotionVectorMagnitudeBuffer)
//{
//    const int blockSize = 256;
//    const int numBlocks = (a_BufferSize + blockSize - 1) / blockSize;
//    SeparateMotionVectorBuffer<<<numBlocks, blockSize>>>(a_BufferSize, a_MotionVectorBuffer, a_MotionVectorDirectionBuffer, a_MotionVectorMagnitudeBuffer);
//
//	cudaDeviceSynchronize();
//}

CPU_ON_GPU void SeparateOptixDenoiserBuffer(
    uint64_t a_BufferSize,
    const float3* a_OptixDenoiserInputBuffer,
    const float3* a_OptixDenoiserAlbedoInputBuffer,
    const float3* a_OptixDenoiserNormalInputBuffer,
    const float3* a_OptixDenoiserOutputBuffer,
    float3* a_OptixDenoiserInputTexture,
    float3* a_OptixDenoiserAlbedoInputTexture,
    float3* a_OptixDenoiserNormalInputTexture,
    float3* a_OptixDenoiserOutputTexture)
{
    const uint32_t bufferSize = a_BufferSize;
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = blockDim.x * gridDim.x;

    for (uint32_t i = index; i < bufferSize - 1; i += stride)
    {
        float3 inputPixel = a_OptixDenoiserInputBuffer[i];
        float3 albedoPixel = a_OptixDenoiserAlbedoInputBuffer[i];
        float3 normalPixel = a_OptixDenoiserNormalInputBuffer[i];

        normalPixel = (normalPixel + 1.f) / 2.f;

        float3 outputPixel = a_OptixDenoiserOutputBuffer[i];;
    	
        a_OptixDenoiserInputTexture[i] = inputPixel;
        a_OptixDenoiserAlbedoInputTexture[i] = albedoPixel;
        a_OptixDenoiserNormalInputTexture[i] = normalPixel;
        a_OptixDenoiserOutputTexture[i] = outputPixel;
    }
}

CPU_ONLY void SeparateOptixDenoiserBufferCPU(
    uint64_t a_BufferSize,
    const float3* a_OptixDenoiserInputBuffer,
    const float3* a_OptixDenoiserAlbedoInputBuffer,
    const float3* a_OptixDenoiserNormalInputBuffer,
    const float3* a_OptixDenoiserOutputBuffer,
    float3* a_OptixDenoiserInputTexture,
    float3* a_OptixDenoiserAlbedoInputTexture,
    float3* a_OptixDenoiserNormalInputTexture,
    float3* a_OptixDenoiserOutputTexture)
{

    const int blockSize = 256;
    const int numBlocks = (a_BufferSize + blockSize - 1) / blockSize;
    SeparateOptixDenoiserBuffer<<<numBlocks, blockSize>>>(
        a_BufferSize, 
        a_OptixDenoiserInputBuffer,
        a_OptixDenoiserAlbedoInputBuffer,
        a_OptixDenoiserNormalInputBuffer,
        a_OptixDenoiserOutputBuffer, 
        a_OptixDenoiserInputTexture,
        a_OptixDenoiserAlbedoInputTexture,
        a_OptixDenoiserNormalInputTexture,
        a_OptixDenoiserOutputTexture);

	cudaDeviceSynchronize();
}
;