#include "SnapShotProcessing.cuh"

#include <device_launch_parameters.h>

CPU_ON_GPU void SeparateIntersectionRayBuffer(WaveFront::AtomicBuffer<WaveFront::IntersectionRayData>* a_IntersectionBuffer,
                                              float3* a_OriginBuffer, float3* a_DirectionBuffer, float3* a_ContributionBuffer)
{
    const uint32_t bufferSize = a_IntersectionBuffer->GetSize();
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = blockDim.x * gridDim.x;

    for (uint32_t i = index; i < bufferSize - 1; i += stride)
    {
        WaveFront::IntersectionRayData* intersectionData = &a_IntersectionBuffer->data[index];

        a_OriginBuffer[intersectionData->m_PixelIndex] = intersectionData->m_Origin;
        a_DirectionBuffer[intersectionData->m_PixelIndex] = intersectionData->m_Direction;
        a_ContributionBuffer[intersectionData->m_PixelIndex] = intersectionData->m_Contribution;
    }
}


CPU_ONLY void SeparateIntersectionRayBufferCPU(uint64_t a_BufferSize, WaveFront::AtomicBuffer<WaveFront::IntersectionRayData>* a_IntersectionBuffer,
    float3* a_OriginBuffer, float3* a_DirectionBuffer, float3* a_ContributionBuffer)
{
    const int blockSize = 256;
    const int numBlocks = (a_BufferSize + blockSize - 1) / blockSize;
    SeparateIntersectionRayBuffer<<<numBlocks, blockSize>>>(a_IntersectionBuffer, a_OriginBuffer, a_DirectionBuffer, a_ContributionBuffer);
};