#include "../Shaders/CppCommon/WaveFrontDataStructs/IntersectionRayBatch.h"


#include <cstdint>
#include "../Shaders/CppCommon/WaveFrontDataStructs/AtomicBuffer.h"

#include "../Shaders/CppCommon/CudaDefines.h"
#include <cuda_runtime.h>

CPU_ON_GPU void SeparateIntersectionRayBuffer(WaveFront::AtomicBuffer<WaveFront::IntersectionRayData>* a_IntersectionBuffer,
    float3* a_OriginBuffer, float3* a_DirectionBuffer, float3* a_ContributionBuffer);

CPU_ONLY void SeparateIntersectionRayBufferCPU(uint64_t a_BufferSize, WaveFront::AtomicBuffer<WaveFront::IntersectionRayData>* a_IntersectionBuffer,
    float3* a_OriginBuffer, float3* a_DirectionBuffer, float3* a_ContributionBuffer);
