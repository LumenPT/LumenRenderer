#pragma once
#include "../Shaders/CppCommon/CudaDefines.h"
#include "../../vendor/Include/sutil/Matrix.h"

namespace WaveFront {
    struct SurfaceData;
}

CPU_ON_GPU void GenerateMotionVector(
    cudaSurfaceObject_t a_MotionVectorBuffer,
    const WaveFront::SurfaceData* a_CurrentSurfaceData,
    uint2 a_Resolution,
    sutil::Matrix4x4* a_PrevViewProjMatrix);