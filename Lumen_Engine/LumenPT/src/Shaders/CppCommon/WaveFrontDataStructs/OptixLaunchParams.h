#pragma once
#include "GPUDataBuffers.h"

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
        IntersectionRayBatch* m_IntersectionRayBatch;
        IntersectionBuffer* m_IntersectionBuffer;
        ShadowRayBatch* m_ShadowRayBatch;
        ResultBuffer* m_ResultBuffer;
        RayType m_TraceType;

    };

}
