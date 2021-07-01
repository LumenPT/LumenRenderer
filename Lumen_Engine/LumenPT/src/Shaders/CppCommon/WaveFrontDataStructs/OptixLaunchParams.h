#pragma once
#include "GPUDataBuffers.h"
#include "AtomicBuffer.h"
#include "../ArrayParameter.h"

#include <Optix/optix.h>
#include <Cuda/cuda/helpers.h>
#include "../SceneDataTableAccessor.h"

namespace WaveFront
{

    enum class  RayType : unsigned int
    {
        INTERSECTION_RAY = 0,
        SHADOW_RAY = 1,
        RESTIR_RAY = 2,
        RESTIR_SHADING_RAY = 3
    };
    
    struct OptixLaunchParameters
    {

        OptixTraversableHandle m_TraversableHandle;
        uint3 m_ResolutionAndDepth;
        AtomicBuffer<IntersectionRayData>* m_IntersectionRayBatch;
        AtomicBuffer<IntersectionData>* m_IntersectionBuffer;
		AtomicBuffer<VolumetricIntersectionData>* m_VolumetricIntersectionBuffer;
        AtomicBuffer<ShadowRayData>* m_ShadowRayBatch;
        AtomicBuffer<RestirShadowRay>* m_ReSTIRShadowRayBatch;
        AtomicBuffer<RestirShadowRayShading>* m_ReSTIRShadowRayShadingBatch;
		SceneDataTableAccessor* m_SceneData;
        Reservoir* m_Reservoirs;
        ArrayParameter<cudaSurfaceObject_t, static_cast<unsigned>(LightChannel::NUM_CHANNELS)> m_OutputChannels;
        float2 m_MinMaxDistance;
        RayType m_TraceType;
    };

}
