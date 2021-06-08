#pragma once
#include "../CudaDefines.h"
#include "PixelIndex.h"
#include "LightData.h"

#include <Optix/optix.h>
#include <Cuda/cuda/helpers.h>
#include <cassert>

namespace WaveFront
{

    struct ShadowRayData
    {

        CPU_GPU ShadowRayData()
            :
        m_PixelIndex({0,0}),
        m_Origin(make_float3(0.f, 0.f, 0.f)),
        m_Direction(make_float3(0.f, 0.f, 0.f)),
        m_MaxDistance(0.f),
        m_PotentialRadiance(make_float3(0.f, 0.f, 0.f)),
        m_OutputChannel(LightChannel::DIRECT)
        {}

        CPU_GPU ShadowRayData(
            const PixelIndex& a_PixelIndex,
            const float3& a_Origin,
            const float3& a_Direction,
            const float& a_MaxDistance,
            const float3& a_PotentialRadiance,
            LightChannel a_OutputChannel)
            :
        m_PixelIndex(a_PixelIndex),
        m_Origin(a_Origin),
        m_Direction(a_Direction),
        m_MaxDistance(a_MaxDistance),
        m_PotentialRadiance(a_PotentialRadiance),
        m_OutputChannel(a_OutputChannel)
        {}

        GPU_ONLY INLINE bool IsValidRay() const
        {

            return
               (m_Direction.x != 0.f ||
                m_Direction.y != 0.f ||
                m_Direction.z != 0.f) &&
                m_MaxDistance > 0.f;

        }



        PixelIndex m_PixelIndex;
        float3 m_Origin;
        float3 m_Direction;
        float m_MaxDistance;
        float3 m_PotentialRadiance;
        LightChannel m_OutputChannel;

    };

}
