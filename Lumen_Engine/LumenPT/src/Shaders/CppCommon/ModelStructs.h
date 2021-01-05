#pragma once

#ifdef LUMEN
#include "glm/gtx/compatibility.hpp"
#define NONAMESPACE glm
typedef unsigned long long cudaTextureObject_t;
#else
#include "Cuda/cuda_runtime.h"
#define NONAMESPACE
#endif

// Common struct for a vertex meant to be used both on the CPU when loading models,
// and on the GPU when reading their data
struct Vertex
{
    NONAMESPACE::float3 m_Position;
    NONAMESPACE::float2 m_UVCoord;
    NONAMESPACE::float3 m_Normal;
    // Can be expanded with additional per-vertex attributes that we need
};

// Common material struct meant as a way to access a model's material on the GPU
struct Material
{
    NONAMESPACE::float4 m_DiffuseColor;
    cudaTextureObject_t m_DiffuseTexture;
};

#undef NONAMESPACE