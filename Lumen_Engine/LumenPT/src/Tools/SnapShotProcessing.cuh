#include "../Shaders/CppCommon/WaveFrontDataStructs/IntersectionRayData.h"


#include <cstdint>
#include "../Shaders/CppCommon/WaveFrontDataStructs/AtomicBuffer.h"
#include "../Shaders/CppCommon/WaveFrontDataStructs/MotionVectorsGenerationData.h"
#include "../Framework/MemoryBuffer.h"

#include "../Shaders/CppCommon/CudaDefines.h"
#include <cuda_runtime.h>



CPU_ON_GPU void SeparateIntersectionRayBuffer(WaveFront::AtomicBuffer<WaveFront::IntersectionRayData>* a_IntersectionBuffer,
    uint2 a_Resolution,
    float3* a_OriginBuffer, float3* a_DirectionBuffer, float3* a_ContributionBuffer);

CPU_ONLY void SeparateIntersectionRayBufferCPU(uint64_t a_BufferSize, WaveFront::AtomicBuffer<WaveFront::IntersectionRayData>* a_IntersectionBuffer,
    uint2 a_Resolution,
    float3* a_OriginBuffer, float3* a_DirectionBuffer, float3* a_ContributionBuffer);

//CPU_ON_GPU void SeparateMotionVectorBuffer(uint64_t a_BufferSize, WaveFront::MotionVectorBuffer* a_MotionVectorBuffer,
//    float3* a_MotionVectorDirectionBuffer, float3* a_MotionVectorMagnitudeBuffer);
//
//CPU_ONLY void SeparateMotionVectorBufferCPU(uint64_t a_BufferSize, WaveFront::MotionVectorBuffer* a_MotionVectorBuffer,
//    float3* a_MotionVectorDirectionBuffer, float3* a_MotionVectorMagnitudeBuffer);

CPU_ON_GPU void SeparateOptixDenoiserBuffer(
    uint64_t a_BufferSize,
    const float3* a_OptixDenoiserInputBuffer,
    const float3* a_OptixDenoiserAlbedoInputBuffer,
    const float3* a_OptixDenoiserNormalInputBuffer,
    const float3* a_OptixDenoiserOutputBuffer,
    float3* a_OptixDenoiserInputTexture,
    float3* a_OptixDenoiserAlbedoInputTexture,
    float3* a_OptixDenoiserNormalInputTexture,
    float3* a_OptixDenoiserOutputTexture);

CPU_ONLY void SeparateOptixDenoiserBufferCPU(
    uint64_t a_BufferSize,
    const float3* a_OptixDenoiserInputBuffer, 
    const float3* a_OptixDenoiserAlbedoInputBuffer,
    const float3* a_OptixDenoiserNormalInputBuffer,
    const float3* a_OptixDenoiserOutputBuffer,
    float3* a_OptixDenoiserInputTexture,
    float3* a_OptixDenoiserAlbedoInputTexture,
    float3* a_OptixDenoiserNormalInputTexture,
    float3* a_OptixDenoiserOutputTexture);