#pragma once
#include "../../Shaders/CppCommon/CudaDefines.h"
#include "../../Shaders/CppCommon/WaveFrontDataStructs.h"

using namespace WaveFront;

CPU_ONLY void ResetIntersectionRayBatch(
    IntersectionRayBatch* const a_RayBatchDevPtr,
    unsigned int a_NumPixels,
    unsigned int a_RaysPerPixel);

CPU_ONLY void ResetShadowRayBatch(
    ShadowRayBatch* a_ShadowRayBatchDevPtr,
    unsigned int a_MaxDepth,
    unsigned int a_NumPixels,
    unsigned int a_RaysPerPixel);

CPU_ONLY void ResetPixelBuffer(
    PixelBuffer* a_PixelBufferDevPtr,
    unsigned int a_NumPixels,
    unsigned int a_ChannelsPerPixel);

CPU_ONLY void ResetLightChannels(
    float3* a_Buffer,
    unsigned int a_NumPixels,
    unsigned int a_ChannelsPerPixel);