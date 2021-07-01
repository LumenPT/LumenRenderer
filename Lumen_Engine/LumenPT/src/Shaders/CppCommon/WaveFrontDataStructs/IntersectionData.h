#pragma once
#include "../CudaDefines.h"
#include "../ModelStructs.h"
#include "IntersectionRayData.h"

#include <Optix/optix.h>
#include <Cuda/cuda/helpers.h>
#include <cassert>

#include "Cuda_fp16.h"

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
            m_InstanceId(0),
            m_PrimitiveIndex(0),
            m_Barycentrics({0.f, 0.f}),
            m_IntersectionT(-1.f)
        {}

        CPU_GPU IntersectionData(
            unsigned int a_RayArrayIndex,
            float a_IntersectionT,
            half2 a_Barycentrics,
            unsigned int a_PrimitiveIndex,
            unsigned int a_InstanceId,
            const PixelIndex& a_PixelIndex)
            :
            m_InstanceId(a_InstanceId),
            m_PrimitiveIndex(a_PrimitiveIndex),
            m_Barycentrics(a_Barycentrics),
            m_IntersectionT(a_IntersectionT)
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
        /// <b>Description</b> \n The unique instance ID of the surface that has been intersected with. \n
        /// <b>Default</b>: 0
        /// </summary>
        unsigned int m_InstanceId;

        /// <summary>
        /// <b>Description</b> \n The index of the primitive(triangle, quad, etc.) of the mesh that the ray intersected with. \n
        /// <b>Default</b>: 0
        /// </summary>
        unsigned int m_PrimitiveIndex;

        /// <summary>
        /// <b>Description</b> \n The U- & V-barycentric coordinates of the intersection point on the triangle that has been interested. \n
        /// <b>Default</b>: 0.f, 0.f
        /// </summary>
        half2 m_Barycentrics;

        /// <summary>
        /// <b>Description</b> \n Distance along the ray the intersection happened. \n
        /// <b>Default</b>: -1.f
        /// </summary>
        float m_IntersectionT;
    };

	union IntersectionDataUint4
	{
        CPU_GPU IntersectionDataUint4()
        {
            m_Data.m_IntersectionT = -1.f;  //Important. If not manually set, junk values are used when not intersecting.
        }
		
        uint4 m_DataAsUint4;
        IntersectionData m_Data;
	};
}
