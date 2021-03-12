#pragma once
#include "../CudaDefines.h"

namespace WaveFront
{

    struct OptixOcclusion
    {

        CPU_GPU OptixOcclusion(float a_MaxDistance)
            :
            m_MaxDistance(a_MaxDistance),
            m_Occluded(false)
        {}

        const float m_MaxDistance;
        bool m_Occluded;

    };

    struct OptixSBTRecordDataRayGen
    {

        float m_MinDistance;
        float m_MaxDistance;

    };

    struct OptixSBTRecordDataMiss
    {
        float m_Dummy;
    };

}
