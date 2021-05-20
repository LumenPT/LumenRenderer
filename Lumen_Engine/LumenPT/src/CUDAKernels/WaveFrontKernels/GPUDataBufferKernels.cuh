#pragma once
#include "../../Shaders/CppCommon/CudaDefines.h"
#include "../../Shaders/CppCommon/WaveFrontDataStructs.h"

using namespace WaveFront;

CPU_ON_GPU void ResetPixelBufferMembers(
    PixelBuffer* const a_PixelBuffer,
    unsigned int a_NumPixels,
    unsigned int a_ChannelsPerPixel);

CPU_ON_GPU void ResetPixelBufferData(PixelBuffer* const a_PixelBuffer);

CPU_ON_GPU void ResetLightChannelData(float3* a_LightData, unsigned a_NumChannels, unsigned a_NumPixels);