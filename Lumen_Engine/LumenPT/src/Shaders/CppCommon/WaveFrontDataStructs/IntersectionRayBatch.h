#pragma once
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
            m_PixelIndex(0),
            m_Origin(make_float3(0.f, 0.f, 0.f)),
            m_Direction(make_float3(0.f, 0.f, 0.f)),
            m_Contribution(make_float3(0.f, 0.f, 0.f))
        {}

        CPU_GPU IntersectionRayData(
            const unsigned a_PixelIndex,
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
        unsigned int m_PixelIndex;
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

    /// <summary>
    /// <b>Description</b> \n
    /// Stores a buffer of IntersectionRayData structs. \n
    /// <b>Type</b>: Struct\n
    /// <para>
    /// <b>Member variables</b> \n
    /// <b>• m_NumPixels</b> <em>(const unsigned int)</em>: Number of pixels the buffer stores data for.\n
    /// <b>• m_RaysPerPixel</b> <em>(const unsigned int)</em>: Number of rays per pixel the buffer stores data for.\n
    /// <b>• m_Rays</b> <em>(RayData[])</em>: Array storing all the RayData structs.\n
    /// </para>
    /// </summary>
    struct IntersectionRayBatch
    {

        IntersectionRayBatch()
            :
            m_NumPixels(0u),
            m_RaysPerPixel(0u),
            m_Rays()
        {}



        /// <summary> Gets the size of the buffer. \n Takes into account the number of pixels and the number of rays per pixel. </summary>
        /// <returns> Size of the buffer.  <em>(unsigned int)</em> </returns>
        CPU_GPU INLINE unsigned int GetSize() const
        {
            return m_NumPixels * m_RaysPerPixel;
        }

        /// <summary> Sets the ray data from a ray for a certain sample for a certain pixel. </summary>
        /// <param name="a_Data">\n • Description: Ray data to set.</param>
        /// <param name="a_PixelIndex">\n • Description: Index of the pixel to set the ray data for. \n • Range: (0 : m_NumPixels-1) </param>
        /// <param name="a_RayIndex">\n • Description: Index of the sample to set the ray data for. \n • Range: (0 : m_RaysPerPixel-1)</param>
        GPU_ONLY INLINE void SetRay(
            const IntersectionRayData& a_Data,
            unsigned int a_PixelIndex,
            unsigned int a_RayIndex = 0)
        {
            m_Rays[GetRayArrayIndex(a_PixelIndex, a_RayIndex)] = a_Data;
        }

        /// <summary> Gets the ray data from a ray for a certain sample for a certain pixel. </summary>
        /// <param name="a_PixelIndex">\n • Description: The index of the pixel to get the ray data from. \n • Range (0 : m_NumPixels-1)</param>
        /// <param name="a_RayIndex">\n • Description: The index of the ray for the pixel. \n • Range: (0 : m_RaysPerPixel-1)</param>
        /// <returns> Data of the specified ray. <em>(const RayData&)</em>  </returns>
        GPU_ONLY INLINE const IntersectionRayData& GetRay(unsigned int a_PixelIndex, unsigned int a_RayIndex) const
        {

            return m_Rays[GetRayArrayIndex(a_PixelIndex, a_RayIndex)];

        }

        /// <summary Gets the ray data from a ray for a certain sample for a certain pixel </summary>
        /// <remarks To get the right index for the pixel and sample you can use the GetRayArrayIndex function </remarks>
        /// <param name="a_RayArrayIndex">\n • Description: The index in the m_Rays array to get the ray data from. \n • Range (0 : (m_NumPixels * m_RaysPerPixel)-1)</param>
        /// <returns> Data of the specified ray. <em>(const RayData&)</em>  </returns>
        GPU_ONLY INLINE const IntersectionRayData& GetRay(unsigned int a_RayArrayIndex) const
        {
            assert(a_RayArrayIndex < GetSize());

            return m_Rays[a_RayArrayIndex];

        }

        /// <summary> Get the index in the m_Rays array for a certain sample at a certain pixel </summary>
        /// <param name="a_PixelIndex">\n • Description: The index of the pixel to get the index for. \n • Range (0 : m_NumPixels-1)</param>
        /// <param name="a_RayIndex">\n • Description: The index of the sample to get the index for. \n • Range (0 : m_RaysPerPixel-1)</param>
        /// <returns> Index into the m_Rays array for the sample at the pixel. <em>(unsigned int)</em>  </returns>
        GPU_ONLY INLINE unsigned int GetRayArrayIndex(unsigned int a_PixelIndex, unsigned int a_RayIndex = 0) const
        {

            assert(a_PixelIndex < m_NumPixels&& a_RayIndex < m_RaysPerPixel);

            return a_PixelIndex * m_RaysPerPixel + a_RayIndex;

        }



        /// <summary>
        /// <b>Description</b> \n The number of pixels the buffer stores data for. \n
        /// <b>Default</b>: 0
        /// </summary>
        const unsigned int m_NumPixels;

        /// <summary>
        /// <b>Description</b> \n  The number of rays per pixel the buffer stores data for. \n
        /// <b>Default</b>: 0
        /// </summary>
        const unsigned int m_RaysPerPixel;

        /// <summary>
        /// <b>Description</b> \n Array storing the RayData structs. \n Has a size of m_NumPixels * m_RaysPerPixel. \n
        /// <b>Default</b>: empty
        /// </summary>
        IntersectionRayData m_Rays[];

    };

}
