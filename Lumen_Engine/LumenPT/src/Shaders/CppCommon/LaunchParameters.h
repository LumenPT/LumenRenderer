#pragma once

#include "ModelStructs.h"

//#include "SceneDataTableAccessor.h"
#include <Optix/optix_types.h>


#include "Cuda/cuda/helpers.h"

class SceneDataTableAccessor;

struct ProgramGroupHeader
{
    unsigned char m_Data[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct LaunchParameters
{
    uchar4* m_Image;
    Vertex* m_VertexBuffer;
    OptixTraversableHandle m_Handle;
    unsigned int m_ImageHeight;
    unsigned int m_ImageWidth;

    SceneDataTableAccessor* m_SceneData;

    float3       eye;
    float3       U;
    float3       V;
    float3       W;
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
