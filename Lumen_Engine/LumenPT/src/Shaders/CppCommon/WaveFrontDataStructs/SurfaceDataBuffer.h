#pragma once
#include "../CudaDefines.h"

#include <Optix/optix.h>
#include <Cuda/cuda/helpers.h>
#include <cassert>

namespace WaveFront
{

    struct SurfaceData
    {

        CPU_GPU SurfaceData(float3 a_Position, 
            float3 a_Normal,
            float a_IntersectionT,
            float3 a_Color,
            float a_Metallic,
            float a_Roughness,
            bool a_Emissive,
            float3 a_IncomingRayDirection,
            float3 a_TransportFactor)
            :
        m_Position(a_Position),
        m_Normal(a_Normal),
        m_IntersectionT(a_IntersectionT),
        m_Color(a_Color),
        m_Metallic(a_Metallic),
        m_Roughness(a_Roughness),
        m_Emissive(a_Emissive),
        m_IncomingRayDirection(a_IncomingRayDirection),
        m_TransportFactor(a_TransportFactor)
        {}

        //Default constructor.
        CPU_GPU SurfaceData() = default;

        //The index of the pixel that this surface data belongs to.
        unsigned m_Index;
        //Position of the intersection in world-space
        float3 m_Position;
        //Normal at the point of intersection.
        float3 m_Normal;
        //Distance along the ray at which the intersection occurs.
        float m_IntersectionT;
        //Direction of the ray that caused the intersection.
        float3 m_IncomingRayDirection;
        //Color at the point of intersection. If m_Emissive is false it is the diffuse color otherwise it is the emissive color.
        float3 m_Color;
        //Metallic factor at the point of intersection.
        float m_Metallic;
        //Roughness factor at the point of intersection.
        float m_Roughness;
        //Defines if the color at the intersection is emissive or diffuse.
        bool m_Emissive;
        //The amount of light that is transported as a scalar-factor.
        float3 m_TransportFactor;

    };

    struct SurfaceDataBuffer
    {

        CPU_GPU SurfaceDataBuffer();

        GPU_ONLY INLINE unsigned int GetSize() const
        {
            return m_NumPixels;
        }

        GPU_ONLY void SetSurfaceData(const SurfaceData& a_Data, unsigned int a_PixelIndex)
        {
            m_SurfaceData[a_PixelIndex] = a_Data;
        }

        GPU_ONLY const SurfaceData& GetSurfaceData(unsigned int a_PixelIndex) const
        {
            assert(a_PixelIndex < GetSize());
            return m_SurfaceData[a_PixelIndex];
        }

        unsigned int m_NumPixels;
        SurfaceData m_SurfaceData[];
    };

}