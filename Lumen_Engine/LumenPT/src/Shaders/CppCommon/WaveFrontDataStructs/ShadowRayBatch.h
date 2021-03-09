#pragma once
#include "../CudaDefines.h"
#include "ResultBuffer.h"

#include <Optix/optix.h>
#include <Cuda/cuda/helpers.h>
#include <cassert>

namespace WaveFront
{

    struct ShadowRayData
    {

        CPU_GPU ShadowRayData()
            :
            m_Origin(make_float3(0.f, 0.f, 0.f)),
            m_Direction(make_float3(0.f, 0.f, 0.f)),
            m_MaxDistance(0.f),
            m_PotentialRadiance(make_float3(0.f, 0.f, 0.f)),
            m_OutputChannelIndex(ResultBuffer::OutputChannel::DIRECT)
        {}

        CPU_GPU ShadowRayData(
            const float3& a_Origin,
            const float3& a_Direction,
            const float& a_MaxDistance,
            const float3& a_PotentialRadiance,
            ResultBuffer::OutputChannel a_OutputChannelIndex)
            :
            m_Origin(a_Origin),
            m_Direction(a_Direction),
            m_MaxDistance(a_MaxDistance),
            m_PotentialRadiance(a_PotentialRadiance),
            m_OutputChannelIndex(a_OutputChannelIndex)
        {}

        GPU_ONLY INLINE bool IsValidRay() const
        {

            return
               (m_Direction.x != 0.f ||
                m_Direction.y != 0.f ||
                m_Direction.z != 0.f) &&
                m_MaxDistance > 0.f;

        }

        //Read only
        float3 m_Origin;
        float3 m_Direction;
        float m_MaxDistance;
        float3 m_PotentialRadiance;
        ResultBuffer::OutputChannel m_OutputChannelIndex;

    };

    struct ShadowRayBatch
    {

        ShadowRayBatch()
            :
            m_MaxDepth(0u),
            m_NumPixels(0u),
            m_RaysPerPixel(0u),
            m_ShadowRays()
        {}

        CPU_GPU INLINE unsigned int GetSize() const
        {
            return m_MaxDepth * m_NumPixels * m_RaysPerPixel;
        }

        GPU_ONLY INLINE void SetShadowRay(
            const ShadowRayData& a_Data,
            unsigned int a_DepthIndex,
            unsigned int a_PixelIndex,
            unsigned int a_RayIndex = 0)
        {
            m_ShadowRays[GetShadowRayArrayIndex(a_DepthIndex, a_PixelIndex, a_RayIndex)] = a_Data;
        }

        GPU_ONLY INLINE const ShadowRayData& GetShadowRayData(unsigned int a_DepthIndex, unsigned int a_PixelIndex, unsigned int a_RayIndex) const
        {

            return m_ShadowRays[GetShadowRayArrayIndex(a_DepthIndex, a_PixelIndex, a_RayIndex)];

        }

        GPU_ONLY INLINE const ShadowRayData& GetShadowRayData(unsigned int a_ShadowRayArrayIndex) const
        {

            assert(a_ShadowRayArrayIndex < GetSize());

            return m_ShadowRays[a_ShadowRayArrayIndex];

        }

        //Gets a index to a ShadowRay in the m_ShadowRays array, taking into account the max dept, number of pixels and number of rays per pixel.
        GPU_ONLY INLINE unsigned int GetShadowRayArrayIndex(unsigned int a_DepthIndex, unsigned int a_PixelIndex, unsigned int a_RayIndex) const
        {

            assert(a_DepthIndex < m_MaxDepth&& a_PixelIndex < m_NumPixels&& a_RayIndex < m_RaysPerPixel);

            return a_DepthIndex * m_NumPixels * m_RaysPerPixel + a_PixelIndex * m_RaysPerPixel + a_RayIndex;

        }

        //Read only
        const unsigned int m_MaxDepth;
        const unsigned int m_NumPixels;
        const unsigned int m_RaysPerPixel;

        //Read/Write
        ShadowRayData m_ShadowRays[];

    };

}
