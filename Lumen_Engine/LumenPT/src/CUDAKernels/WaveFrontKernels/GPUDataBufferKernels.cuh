#pragma once
#include "../../Shaders/CppCommon/CudaDefines.h"
#include "../../Shaders/CppCommon/WaveFrontDataStructs.h"

using namespace WaveFront;

CPU_ON_GPU void ResetIntersectionRayBatchMembers(
    IntersectionRayBatch* const a_RayBatch,
    unsigned int a_NumPixels,
    unsigned int a_RaysPerPixel);

CPU_ON_GPU void ResetIntersectionRayBatchData(IntersectionRayBatch* const a_RayBatch);

CPU_ON_GPU void ResetShadowRayBatchMembers(
    ShadowRayBatch* const a_ShadowRayBatch,
    unsigned int a_MaxDepth,
    unsigned int a_NumPixels,
    unsigned int a_RaysPerPixel);

CPU_ON_GPU void ResetShadowRayBatchData(ShadowRayBatch* const a_ShadowRayBatch);

CPU_ON_GPU void ResetPixelBufferMembers(
    PixelBuffer* const a_PixelBuffer,
    unsigned int a_NumPixels,
    unsigned int a_ChannelsPerPixel);

CPU_ON_GPU void ResetPixelBufferData(PixelBuffer* const a_PixelBuffer);

CPU_ON_GPU void ResetLightChannelData(float3* a_LightData, unsigned a_NumChannels, unsigned a_NumPixels);