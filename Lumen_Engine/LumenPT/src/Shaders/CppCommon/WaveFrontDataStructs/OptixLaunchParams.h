#pragma once
#include "GPUDataBuffers.h"
#include "AtomicBuffer.h"

#include <Optix/optix.h>
#include <Cuda/cuda/helpers.h>

namespace WaveFront
{

    enum class  RayType : unsigned int
    {
        INTERSECTION_RAY = 0,
        SHADOW_RAY = 1,
        RESTIR_RAY = 2
    };
    
    struct OptixLaunchParameters
    {

        OptixTraversableHandle m_TraversableHandle;
        uint3 m_ResolutionAndDepth;
        AtomicBuffer<IntersectionRayData>* m_IntersectionRayBatch;
        AtomicBuffer<IntersectionData>* m_IntersectionBuffer;
        AtomicBuffer<ShadowRayData>* m_ShadowRayBatch;
        AtomicBuffer<RestirShadowRay>* m_ReSTIRShadowRayBatch;
        Reservoir* m_Reservoirs;
        float3* m_ResultBuffer;
        float2 m_MinMaxDistance;
        RayType m_TraceType;
    };

}
