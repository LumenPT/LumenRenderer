#pragma once
#include "../../Shaders/CppCommon/CudaDefines.h"
#include "../../Shaders/CppCommon/WaveFrontDataStructs.h"

using namespace WaveFront;

CPU_ONLY void ResetPixelBuffer(
    PixelBuffer* a_PixelBufferDevPtr,
    unsigned int a_NumPixels,
    unsigned int a_ChannelsPerPixel);

CPU_ONLY void ResetLightChannels(
    float3* a_Buffer,
    unsigned int a_NumPixels,
    unsigned int a_ChannelsPerPixel);