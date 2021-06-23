#pragma once
#include "../../Shaders/CppCommon/CudaDefines.h"
#include "../../Shaders/CppCommon/WaveFrontDataStructs.h"
#include "../../Shaders/CppCommon/WaveFrontDataStructs/PixelIndex.h"

class SceneDataTableAccessor;

CPU_ON_GPU void ExtractVolumetricDataGpu(
    unsigned a_NumIntersections,
    const WaveFront::AtomicBuffer<WaveFront::IntersectionRayData>* a_Rays,
    const WaveFront::AtomicBuffer<WaveFront::VolumetricIntersectionData>* a_IntersectionData,
    WaveFront::VolumetricData* a_OutPut,
    uint2 a_Resolution,
    SceneDataTableAccessor* a_SceneDataTable);

GPU_ONLY void VolumetricShadeDirect(
    WaveFront::PixelIndex a_PixelIndex,
    const uint3 a_ResolutionAndDepth,
    const WaveFront::VolumetricData* a_VolumetricDataBuffer,
    WaveFront::AtomicBuffer<WaveFront::ShadowRayData>* const a_ShadowRays,
    const WaveFront::AtomicBuffer<WaveFront::TriangleLight>* const a_Lights,
	unsigned int seed,
    const CDF* const a_CDF = nullptr,
	cudaSurfaceObject_t a_Output = 0);