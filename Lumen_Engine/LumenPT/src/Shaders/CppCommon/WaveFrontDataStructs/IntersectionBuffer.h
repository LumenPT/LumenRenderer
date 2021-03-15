#pragma once
#include "../CudaDefines.h"
#include "../ModelStructs.h"
#include "IntersectionRayBatch.h"

#include <Optix/optix.h>
#include <Cuda/cuda/helpers.h>
#include <cassert>

namespace WaveFront
{

/// <summary>
/// <b>Description</b> \n
/// Stores data of an intersection. \n
/// <b>Type</b>: Struct\n
/// <para>
/// <b>Member variables</b> \n
/// <b>• m_RayIndex</b> <em>(unsigned int)</em>: Index in the m_Rays member of a RayBatch of the ray the intersection belongs to.\n
/// <b>• m_IntersectionT</b> <em>(float)</em>: Distance along the ray to the intersection.\n
/// <b>• m_PrimitiveIndex</b> <em>(unsigned int): Index of the primitive of the mesh intersected by the ray.</em>: .\n
/// </para>
/// </summary>
    struct IntersectionData
    {

        CPU_GPU IntersectionData()
            :
            m_RayArrayIndex(0),
            m_IntersectionT(-1.f),
            m_Barycentrics({0.f, 0.f}),
            m_PrimitiveIndex(0),
            m_InstanceId(0),
            m_PixelIndex(0)
        {}

        CPU_GPU IntersectionData(
            unsigned int a_RayArrayIndex,
            float a_IntersectionT,
            float2 a_Barycentrics,
            unsigned int a_PrimitiveIndex,
            unsigned int a_InstanceId,
            unsigned int a_PixelIndex)
            :
            m_RayArrayIndex(a_RayArrayIndex),
            m_IntersectionT(a_IntersectionT),
            m_Barycentrics(a_Barycentrics),
            m_PrimitiveIndex(a_PrimitiveIndex),
            m_InstanceId(a_InstanceId),
            m_PixelIndex(a_PixelIndex)
        {}



        /// <summary> Checks if the data defines an intersection. </summary>
        /// <returns> Returns true if m_IntersectionT is higher than 0.  <em>(boolean)</em> </returns>
        CPU_GPU INLINE bool IsIntersection() const
        {
            return (m_IntersectionT > 0.f);
        }

        /// <summary> Calculates the intersection point in world-space when given the right ray data as input.
        /// If this intersection data is not an intersection, returns a 0 vector.
        /// The calculation done is simply</summary>
        /// <param name="a_RayData">\n • Description: A reference to the RayData struct used to construct the ray.</param>
        /// <returns>Intersection in world-space if the intersection data is an intersection, else returns a 0 vector.</returns>
        CPU_GPU INLINE float3 GetIntersectionPoint(const IntersectionRayData& a_RayData) const
        {
            if (IsIntersection())
            {
                return a_RayData.m_Origin + m_IntersectionT * a_RayData.m_Direction;
            }
            else return make_float3(0.f);
        }



        /// <summary>
        /// <b>Description</b> \n The index in the m_Rays array of a RayBatch of the ray the intersection belongs to. \n
        /// <b>Default</b>: 0
        /// </summary>
        unsigned int m_RayArrayIndex;

        //The index of the pixel/surface that this intersection affects.
        unsigned int m_PixelIndex;

        //Instance ID unique to the surface intersected.
        unsigned int m_InstanceId;

        //The barycentric coordinates of the triangle hit.
        float2 m_Barycentrics;

        /// <summary>
        /// <b>Description</b> \n The index of the primitive(triangle, quad, etc.) of the mesh that the ray intersected with. \n
        /// <b>Default</b>: 0
        /// </summary>
        unsigned int m_PrimitiveIndex;

        /// <summary>
        /// <b>Description</b> \n Distance along the ray the intersection happened. \n
        /// <b>Default</b>: -1.f
        /// </summary>
        float m_IntersectionT;
    };

    struct IntersectionBuffer
    {

        IntersectionBuffer()
            :
            m_NumPixels(0u),
            m_IntersectionsPerPixel(0u),
            m_Intersections()
        {}



        CPU_GPU INLINE unsigned int GetSize() const
        {
            return m_NumPixels * m_IntersectionsPerPixel;
        }

        GPU_ONLY INLINE void SetIntersection(
            const IntersectionData& a_Data,
            unsigned int a_PixelIndex,
            unsigned int a_IntersectionIndex = 0)
        {

            m_Intersections[GetIntersectionArrayIndex(a_PixelIndex, a_IntersectionIndex)] = a_Data;

        }

        GPU_ONLY INLINE const IntersectionData& GetIntersection(unsigned int a_PixelIndex, unsigned int a_IntersectionIndex) const
        {

            return m_Intersections[GetIntersectionArrayIndex(a_PixelIndex, a_IntersectionIndex)];

        }

        GPU_ONLY INLINE const IntersectionData& GetIntersection(unsigned int a_IntersectionArrayIndex) const
        {
            assert(a_IntersectionArrayIndex < GetSize());

            return m_Intersections[a_IntersectionArrayIndex];

        }

        //Gets a index to IntersectionData in the m_Intersections array, taking into account the number of pixels and the number of rays per pixel.
        GPU_ONLY INLINE unsigned int GetIntersectionArrayIndex(unsigned int a_PixelIndex, unsigned int a_IntersectionIndex = 0) const
        {

            assert(a_PixelIndex < m_NumPixels && a_IntersectionIndex < m_IntersectionsPerPixel);

            return a_PixelIndex * m_IntersectionsPerPixel + a_IntersectionIndex;

        }



        //Read only
        const unsigned int m_NumPixels;
        const unsigned int m_IntersectionsPerPixel;

        //Read/Write
        IntersectionData m_Intersections[];

    };



}