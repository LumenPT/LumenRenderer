#pragma once

#include "Cuda/cuda/helpers.h"

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
    
};

struct HitData
{
    
};
