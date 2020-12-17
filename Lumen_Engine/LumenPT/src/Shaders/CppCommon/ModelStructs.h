#pragma once

#include "Cuda/cuda_runtime.h"

// Common struct for a vertex meant to be used both on the CPU when loading models,
// and on the GPU when reading their data
struct Vertex
{
    float3 m_Position;
    float2 m_UVCoord;
    float3 m_Normal;
    // Can be expanded with additional per-vertex attributes that we need
};

// Common material struct meant as a way to access a model's material on the GPU
struct Material
{
    float4 m_DiffuseColor;
    cudaTextureObject_t m_DiffuseTexture;
};