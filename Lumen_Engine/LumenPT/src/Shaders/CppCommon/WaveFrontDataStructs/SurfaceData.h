#pragma once
#include "../CudaDefines.h"

#include <Optix/optix.h>
#include <Cuda/cuda/helpers.h>
#include <cassert>

#include "../ShadingData.h"

namespace WaveFront
{

    struct SurfaceData
    {

        CPU_GPU SurfaceData(float3 a_Position, 
            float3 a_Normal,
            float a_IntersectionT,
            ShadingData a_ShadingData,
            bool a_Emissive,
            float3 a_IncomingRayDirection,
            float3 a_TransportFactor)
            :
        m_Position(a_Position),
        m_Normal(a_Normal),
        m_IntersectionT(a_IntersectionT),
    	m_ShadingData(a_ShadingData),
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
        //Tangent at the point of intersection.
        float3 m_Tangent;
        //Distance along the ray at which the intersection occurs.
        float m_IntersectionT;
        //Direction of the ray that caused the intersection.
        float3 m_IncomingRayDirection;

    	//Shading related data such as color, roughness, metallicness and transmission. etc.
        ShadingData m_ShadingData;  
    	
        //Defines if the color at the intersection is emissive or diffuse.
        bool m_Emissive;
        //The amount of light that is transported as a scalar-factor.
        float3 m_TransportFactor;

    };

}
