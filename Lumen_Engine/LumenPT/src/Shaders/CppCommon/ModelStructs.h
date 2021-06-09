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

namespace Lumen {
    enum class EmissionMode;
}

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
        : m_TransmissionTexture(0), m_TransmissionFactor(0), m_ClearCoatTexture(0), m_ClearCoatRoughnessTexture(0),
          m_ClearCoatFactor(0),
          m_ClearCoatRoughnessFactor(0),
          m_IndexOfRefraction(0),
          m_SpecularFactor(0),
          m_SpecularTintFactor(0),
          m_SubSurfaceFactor(0), m_Luminance(0), m_Anisotropic(0), m_SheenFactor(0.f),
          m_TintFactor(make_float3(0.f, 0.f, 0.f)), m_TransmittanceFactor(make_float3(0.f, 0.f, 0.f)),
          m_DiffuseColor({0.f, 0.f, 0.f, 0.f}),
          m_EmissionColor({0.f, 0.f, 0.f, 0.f}),
          m_DiffuseTexture(0),
          m_EmissiveTexture(0),
          m_MetalRoughnessTexture(0),
          m_NormalTexture(0), m_SheenTintFactor(0), m_TintTexture(0)
    {
    }

    //TODO: pack these together.

    //Transmission
    cudaTextureObject_t m_TransmissionTexture;
    float m_TransmissionFactor;

    //Clearcoat
    cudaTextureObject_t m_ClearCoatTexture;
    cudaTextureObject_t m_ClearCoatRoughnessTexture;
    float m_ClearCoatFactor;
    float m_ClearCoatRoughnessFactor;

    //Refraction
    float m_IndexOfRefraction;

    //Specular
    float m_SpecularFactor;
    float m_SpecularTintFactor;

    //Subsurface scattering. //NOTE: This is not provided by GLTF at the moment. 
    float m_SubSurfaceFactor;

    float m_Luminance;
    float m_Anisotropic;

    float m_MetallicFactor;
    float m_RoughnessFactor;

    float m_SheenTintFactor;
    float m_SheenFactor;
    float3 m_TintFactor;
    float3 m_TransmittanceFactor;


    NONAMESPACE::float4 m_DiffuseColor;
    NONAMESPACE::float4 m_EmissionColor;
    cudaTextureObject_t m_DiffuseTexture;
    cudaTextureObject_t m_EmissiveTexture;
    cudaTextureObject_t m_MetalRoughnessTexture;
    cudaTextureObject_t m_NormalTexture;
    cudaTextureObject_t m_TintTexture;
};

//TODO: change this naming because it is confusing, it could be name DevicePrimitiveArray or DeviceMesh
struct DevicePrimitive
{
    Vertex*         m_VertexBuffer;
    unsigned int*   m_IndexBuffer;
    DeviceMaterial* m_Material;
    bool*           m_IsEmissive;
};

struct DevicePrimitiveInstance
{
    DevicePrimitive m_Primitive;
    // Other instance-specific data
    sutil::Matrix4x4 m_Transform;       //The transform for this primitive.
    Lumen::EmissionMode m_EmissionMode; //The emission mode for this primitive.
    float4 m_EmissiveColorAndScale;     //RGB color override and W = scaling factor.
};

//TODO remove old
///*
// * A struct on the GPU containing pointers to the data buffers for a specific mesh.
// */
//struct DeviceMesh
//{
//    Vertex*         m_VertexBuffer;	//The vertex buffer.
//    unsigned int*   m_IndexBuffer;	//The index buffer.
//};
//
///*
// * Instance data on the GPU about a particular scene object.
// */
//struct DeviceInstanceData
//{
//    DeviceMesh*         m_Mesh;			//Pointer to the mesh data for this instance.
//    DeviceMaterial*     m_Material;		//The material for this instance.
//    sutil::Matrix4x4	m_Transform;	//The transform for this instance.
//    float3				m_Radiance;		//The emissive radiance for this instance.
//    bool				m_IsEmissive;	//True when emissive, false when not.
//};

#undef NONAMESPACE