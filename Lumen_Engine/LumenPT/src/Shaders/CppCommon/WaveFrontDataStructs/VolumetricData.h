#pragma once
#include "../CudaDefines.h"

#include <Optix/optix.h>
#include <Cuda/cuda/helpers.h>
#include <cassert>

namespace WaveFront
{

    struct VolumetricData
    {

        PixelIndex m_PixelIndex;

        float3 m_PositionEntry;

        float3 m_PositionExit;

        float m_EntryIntersectionT;

        float m_ExitIntersectionT;

        float3 m_IncomingRayDirection;

		const nanovdb::FloatGrid* m_VolumeGrid;

    };

}
