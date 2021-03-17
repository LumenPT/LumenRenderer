#pragma once
#include "../../Shaders/CppCommon/CudaDefines.h"
#include "../../Shaders/CppCommon/WaveFrontDataStructs.h"

using namespace WaveFront;

class SceneDataTableAccessor;

#define GET_PIXEL_INDEX(X, Y, WIDTH) ((Y) * WIDTH + (X))

GPU_ONLY void HaltonSequence(
    unsigned int index,
    unsigned int base,
    float* result);

//Generate some rays based on the thread index.
CPU_ON_GPU void GeneratePrimaryRay(
    int a_NumRays,
    AtomicBuffer<IntersectionRayData>* const a_Buffer,
    float3 a_U,
    float3 a_V,
    float3 a_W,
    float3 a_Eye,
    int2 a_Dimensions,
    unsigned int a_FrameCount);

CPU_ON_GPU void ExtractSurfaceDataGpu(unsigned a_NumIntersections,
    AtomicBuffer<IntersectionData>* a_IntersectionData,
    AtomicBuffer<IntersectionRayData>* a_Rays,
    SurfaceData* a_OutPut,
    SceneDataTableAccessor* a_SceneDataTable);

//Called during shading

/*
 * Called at the start of the Shade function with a number of thread-blocks.
 * Calculates direct shading as potential light contribution.
 * Defines a shadow ray that is used to validate the potential light contribution.
 */
CPU_ON_GPU void ShadeDirect(
    const uint3 a_ResolutionAndDepth,
    const IntersectionRayBatch* const a_CurrentRays,
    const IntersectionBuffer* const a_CurrentIntersections,
    ShadowRayBatch* const a_ShadowRays,
    const LightDataBuffer* const a_Lights,
    CDF* const a_CDF /*const CDF* a_CDF*/);

/*
 *
 */
CPU_ON_GPU void ShadeSpecular();

/*
 *
 */
CPU_ON_GPU void ShadeIndirect(
    const uint3 a_ResolutionAndDepth,
    const IntersectionRayBatch* const a_CurrentRays,
    const IntersectionBuffer* const a_Intersections,
    IntersectionRayBatch* const a_Output);


CPU_ON_GPU void DEBUGShadePrimIntersections(
    const uint3 a_ResolutionAndDepth,
    const SurfaceData* a_CurrentSurfaceData,
    float3* const a_Output
);

//Called during post-processing.

/*
 *
 */
CPU_ON_GPU void Denoise();

/*
 *
 */
CPU_ON_GPU void MergeOutputChannels(
    const uint2 a_Resolution,
    const float3* const a_Input,
    float3* const a_Output);

/*
 *
 */
CPU_ON_GPU void DLSS();

/*
 *
 */
CPU_ON_GPU void PostProcessingEffects();

//Temporary step till post-processing is in place.
CPU_ON_GPU void WriteToOutput(
    const uint2 a_Resolution,
    const float3* const a_Input,
    uchar4* a_Output);