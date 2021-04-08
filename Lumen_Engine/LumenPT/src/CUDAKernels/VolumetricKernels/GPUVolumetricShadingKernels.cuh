#pragma once
#include "../../Shaders/CppCommon/CudaDefines.h"
#include "../../Shaders/CppCommon/WaveFrontDataStructs.h"

class SceneDataTableAccessor;

CPU_ON_GPU void ExtractVolumetricDataGpu(
    unsigned a_NumIntersections,
    const WaveFront::AtomicBuffer<WaveFront::IntersectionRayData>* a_Rays,
    const WaveFront::AtomicBuffer<WaveFront::VolumetricIntersectionData>* a_IntersectionData,
    WaveFront::VolumetricData* a_OutPut,
    SceneDataTableAccessor* a_SceneDataTable);

GPU_ONLY void VolumetricShadeDirect(
	unsigned int a_PixelIndex,
    const uint3 a_ResolutionAndDepth,
    const WaveFront::VolumetricData* a_VolumetricDataBuffer,
    WaveFront::AtomicBuffer<WaveFront::ShadowRayData>* const a_ShadowRays,
    const WaveFront::TriangleLight* const a_Lights,
    const unsigned int a_NumLights,
    const CDF* const a_CDF = nullptr,
	float3* a_Output = nullptr);