#pragma once

#ifdef LUMEN
#include "glm/gtx/compatibility.hpp"
#define NONAMESPACE glm
typedef unsigned long long cudaTextureObject_t;
#else
#include "Cuda/cuda_runtime.h"
#define NONAMESPACE
#endif
#include <sutil/Matrix.h>

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

    DeviceMaterial()
        :
    m_DiffuseColor({0.f, 0.f, 0.f, 0.f}),
    m_EmissionColor({0.f, 0.f, 0.f, 0.f}),
    m_DiffuseTexture(0),
    m_EmissiveTexture(0),
    m_MetalRoughnessTexture(0),
    m_NormalTexture(0)
    {}

    NONAMESPACE::float4 m_DiffuseColor;
    NONAMESPACE::float4 m_EmissionColor;
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

/*
 * A struct on the GPU containing pointers to the data buffers for a specific mesh.
 */
struct DeviceMesh
{
    Vertex*         m_VertexBuffer;	//The vertex buffer.
    unsigned int*   m_IndexBuffer;	//The index buffer.
};

/*
 * Instance data on the GPU about a particular scene object.
 */
struct DeviceInstanceData
{
    DeviceMesh*         m_Mesh;			//Pointer to the mesh data for this instance.
    DeviceMaterial*     m_Material;		//The material for this instance.
    sutil::Matrix4x4	m_Transform;	//The transform for this instance.
    float3				m_Radiance;		//The emissive radiance for this instance.
    bool				m_IsEmissive;	//True when emissive, false when not.
};

#undef NONAMESPACE