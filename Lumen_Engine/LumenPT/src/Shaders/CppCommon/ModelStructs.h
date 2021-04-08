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
    NONAMESPACE::float4 m_Tangent;
    // Can be expanded with additional per-vertex attributes that we need
};


// Common material struct meant as a way to access a model's material on the GPU
struct DeviceMaterial
{
    NONAMESPACE::float4 m_DiffuseColor;
    NONAMESPACE::float3 m_EmissionColor;
    cudaTextureObject_t m_DiffuseTexture;
    cudaTextureObject_t m_EmissiveTexture;
    cudaTextureObject_t m_MetalRoughnessTexture;
    cudaTextureObject_t m_NormalTexture;
};

//TODO: change this naming because it is confusing, it could be name DevicePrimitiveArray or DeviceMesh
struct DevicePrimitive
{
    Vertex*         m_VertexBuffer;
    unsigned int*   m_IndexBuffer;
    DeviceMaterial* m_Material;
    bool*           m_IsEmissive;
};

#undef NONAMESPACE