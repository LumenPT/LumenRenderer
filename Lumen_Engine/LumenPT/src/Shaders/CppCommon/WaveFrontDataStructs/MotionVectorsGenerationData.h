#pragma once
#include "../CudaDefines.h"
#include "../WaveFrontDataStructs.h"

#include <Optix/optix.h>
#include <Cuda/cuda/helpers.h>
#include <sutil/Matrix.h>
#include <cassert>

namespace WaveFront
{
	
    struct MotionVectorsGenerationData
    {
        cudaSurfaceObject_t m_MotionVectorBuffer;
		const SurfaceData* m_CurrentSurfaceData;
        uint2 m_RenderResolution;
		sutil::Matrix4x4 m_PrevViewMatrix;
		sutil::Matrix4x4 m_ProjectionMatrix;
    };

}