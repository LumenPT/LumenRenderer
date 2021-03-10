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
            float a_Depth,
            float3 a_DiffuseColor,
            float a_Metallic,
            float a_Roughness,
            float3 a_EmissiveColor,
            float3 a_IncomingRay,
            float3 a_TransportFactor)
            :
        m_Position(a_Position),
        m_Normal(a_Normal),
        m_Depth(a_Depth),
        m_DiffuseColor(a_DiffuseColor),
        m_Metallic(a_Metallic),
        m_Roughness(a_Roughness),
        m_EmissiveColor(a_EmissiveColor),
        m_IncomingRay(a_IncomingRay),
        m_TransportFactor(a_TransportFactor)
        {}

        float3 m_Position;
        float3 m_Normal;
        float m_Depth;
        float3 m_DiffuseColor;
        float m_Metallic;
        float m_Roughness;
        float3 m_EmissiveColor;
        float3 m_IncomingRay;
        float3 m_TransportFactor;

    };

    struct SurfaceDataBuffer
    {

        CPU_GPU SurfaceDataBuffer();

        GPU_ONLY INLINE unsigned int GetSize() const
        {

            return m_NumPixels * m_NumSurfaceDataPerPixel;

        }

        GPU_ONLY void SetSurfaceData(const SurfaceData& a_Data, unsigned int a_PixelIndex, unsigned int a_DataIndex)
        {

            m_SurfaceData[GetSurfaceDataArrayIndex(a_PixelIndex, a_DataIndex)] = a_Data;

        }

        GPU_ONLY const SurfaceData& GetSurfaceData(unsigned int a_PixelIndex, unsigned int a_DataIndex) const
        {

            return m_SurfaceData[GetSurfaceDataArrayIndex(a_PixelIndex, a_DataIndex)];

        }

        GPU_ONLY const SurfaceData& GetSurfaceData(unsigned int a_SurfaceDataArrayIndex) const
        {

            assert(a_SurfaceDataArrayIndex < GetSize());

            return m_SurfaceData[a_SurfaceDataArrayIndex];

        }

        GPU_ONLY unsigned int GetSurfaceDataArrayIndex(unsigned int a_PixelIndex, unsigned int a_DataIndex) const
        {

            assert(a_PixelIndex < m_NumPixels && a_DataIndex < m_NumSurfaceDataPerPixel);

            return a_PixelIndex * m_NumSurfaceDataPerPixel + a_DataIndex;

        }

        unsigned int m_NumPixels;
        unsigned int m_NumSurfaceDataPerPixel;

        SurfaceData m_SurfaceData[];

    };

}