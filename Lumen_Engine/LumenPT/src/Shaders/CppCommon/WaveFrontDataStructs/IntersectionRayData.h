#pragma once
#include "PixelIndex.h"
#include "../CudaDefines.h"

#include <Optix/optix.h>
#include <Cuda/cuda/helpers.h>
#include <cassert>

namespace WaveFront
{

/// <summary>
/// <b>Description</b> \n
/// Stores definition of a intersection ray and additional data.\n
/// <b>Type</b>: Struct\n
/// <para>
/// <b>Member variables:</b> \n
/// <b>• m_Origin</b> <em>(float3)</em>: Origin of the ray. \n
/// <b>• m_Direction</b> <em>(float3)</em>: Direction of the ray. \n
/// <b>• m_Contribution</b> <em>(float3)</em>: Contribution scalar for the returned radiance. \n
/// </para>
/// </summary>
    struct IntersectionRayData
    {

        CPU_GPU IntersectionRayData()
            :
            m_PixelIndex({0, 0}),
            m_Origin(make_float3(0.f, 0.f, 0.f)),
            m_Direction(make_float3(0.f, 0.f, 0.f)),
            m_Contribution(make_float3(0.f, 0.f, 0.f))
        {}

        CPU_GPU IntersectionRayData(
            const PixelIndex& a_PixelIndex,
            const float3& a_Origin,
            const float3& a_Direction,
            const float3& a_Contribution
            )
            :
            m_PixelIndex(a_PixelIndex),
            m_Origin(a_Origin),
            m_Direction(a_Direction),
            m_Contribution(a_Contribution)
        {}



        /// <summary> Checks if the ray is a valid ray.</summary>
        /// <returns> Returns true if <b>all</b> of the components of m_Direction are not equal to 0. <em>(boolean)</em> </returns>
        GPU_ONLY INLINE bool IsValidRay() const
        {

            return (m_Direction.x != 0.f ||
                m_Direction.y != 0.f ||
                m_Direction.z != 0.f);

        }


        //The index of the pixel that this ray contributes to.
        PixelIndex m_PixelIndex;
        /// <summary>
        /// <b>Description</b> \n Stores the position of the ray interpreted as a world-space position. \n
        /// <b>Default</b>: (0.f, 0.f, 0.f)
        /// </summary>
        float3 m_Origin;
        /// <summary>
        /// <b>Description</b> \n Stores the direction of the ray interpreted as a normalized vector. \n
        /// <b>Default</b>: (0.f, 0.f, 0.f)
        /// </summary>
        float3 m_Direction;
        /// <summary>
        /// <b>Description</b> \n Stores the contribution of the radiance returned by the ray as a scalar for each rgb-channel. \n
        /// <b>Default</b>: (0.f, 0.f, 0.f)
        /// </summary>
        float3 m_Contribution;

    };

}
