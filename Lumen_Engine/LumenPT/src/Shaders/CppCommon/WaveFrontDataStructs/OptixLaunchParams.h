#pragma once
#include "GPUDataBuffers.h"
#include "SceneDataTable.h"
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
        float3* m_ResultBuffer;
        RayType m_TraceType;
    };

}
