#pragma once

#include <Optix/optix_types.h>

#include "Cuda/cuda/helpers.h"


struct ProgramGroupHeader
{
    unsigned char m_Data[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct LaunchParameters
{
    uchar4* m_Image;
    OptixTraversableHandle m_Handle;
    unsigned int m_ImageHeight;
    unsigned int m_ImageWidth;
};

struct RaygenData
{
    float3 m_Color;
};

struct MissData
{
    unsigned int m_Num;
    float3 m_Color;
};

struct HitData
{
    cudaTextureObject_t m_TextureObject;
};
