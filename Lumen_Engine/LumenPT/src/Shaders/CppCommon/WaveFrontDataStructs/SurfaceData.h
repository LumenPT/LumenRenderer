#pragma once
#include "../CudaDefines.h"
#include "PixelIndex.h"

#include <Optix/optix.h>
#include <Cuda/cuda/helpers.h>
#include <cassert>

#include "../ShadingData.h"

namespace WaveFront
{

	/*
	 * Surface flags for quick bit comparisons. 
	 */
    enum SurfaceFlags : unsigned char
    {
    	SURFACE_FLAG_NONE = 0,
        SURFACE_FLAG_EMISSIVE = 1 << 0,
        SURFACE_FLAG_ALPHA_TRANSPARENT = 1 << 1,
        SURFACE_FLAG_NON_INTERSECT = 1 << 2,
	};

	__device__ __forceinline__ SurfaceFlags operator | (const SurfaceFlags& a_Lhs, const SurfaceFlags& a_Rhs)
	{
        return static_cast<SurfaceFlags>(static_cast<unsigned char>(a_Lhs) | static_cast<unsigned char>(a_Rhs));
	}
	
    __device__ __forceinline__ SurfaceFlags operator & (const SurfaceFlags& a_Lhs, const SurfaceFlags& a_Rhs)
    {
        return static_cast<SurfaceFlags>(static_cast<unsigned char>(a_Lhs) & static_cast<unsigned char>(a_Rhs));
    }
	
    __device__ __forceinline__ SurfaceFlags operator |= (SurfaceFlags& a_Lhs, const SurfaceFlags& a_Rhs)
    {
        return a_Lhs = static_cast<SurfaceFlags>(a_Lhs | a_Rhs);
    }

    __device__ __forceinline__ SurfaceFlags operator ~ (const SurfaceFlags& a_Lhs)
    {
        return static_cast<SurfaceFlags>(~static_cast<unsigned char>(a_Lhs));
    }

	/*
	 * Describes an intersected surface.
	 * Contains all information required for shading.
	 */
    struct SurfaceData
    {

        CPU_GPU SurfaceData(float3 a_Position,
            float3 a_Normal,
            float3 a_GeometricNormal,
            float a_IntersectionT,
            MaterialData a_MaterialData,
            SurfaceFlags a_Surface_flags,
            float3 a_IncomingRayDirection,
            float3 a_TransportFactor)
	        : m_Position(a_Position),
	          m_Normal(a_Normal),
	          m_GeometricNormal(a_GeometricNormal), m_Tangent(),
	          m_IntersectionT(a_IntersectionT),
	          m_MaterialData(a_MaterialData),
	          m_SurfaceFlags(a_Surface_flags),
	          m_IncomingRayDirection(a_IncomingRayDirection),
	          m_TransportFactor(a_TransportFactor)
        {
        }

        //Default constructor.
        CPU_GPU SurfaceData() = default;

        //The index of the pixel that this surface data belongs to.
        PixelIndex m_PixelIndex;

        //Position of the intersection in world-space
        float3 m_Position;

        //Normal at the point of intersection.
        //m_Normal = the normal after interpolation and normal mapping.
        //m_GeometricNormal = the plain normal without fancy effects.
        float3 m_Normal;
        float3 m_GeometricNormal;

        //Tangent at the point of intersection.
        float3 m_Tangent;

        //Distance along the ray at which the intersection occurs.
        float m_IntersectionT;

        //Direction of the ray that caused the intersection.
        float3 m_IncomingRayDirection;

    	//Shading related data such as color, roughness, metallicness and transmission. etc.
        MaterialData m_MaterialData;  
    	
        //Flags telling whether or not this surface is emissive, intersected or alpha transparent.
        SurfaceFlags m_SurfaceFlags;

        //The amount of light that is transported as a scalar-factor.
        float3 m_TransportFactor;

    };

}
