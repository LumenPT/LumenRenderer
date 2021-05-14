#pragma once
#include "../../Shaders/CppCommon/CudaDefines.h"
#include "../../Shaders/CppCommon/WaveFrontDataStructs/CudaKernelParamStructs.h"

class SceneDataTableAccessor;

//CPU_GPU void HaltonSequence(
//    int index,
//    int base,
//    float* result);

/*
 * Called at start of frame.
 * Generates the camera rays.
 * Synchronizes with device at the end of the function.
 */
CPU_ONLY void GeneratePrimaryRays(const WaveFront::PrimRayGenLaunchParameters& a_PrimRayGenParams);

/*
 *
 */
CPU_ONLY void GenerateMotionVectors(WaveFront::MotionVectorsGenerationData& a_MotionVectorsData);

/*
 * Extract the surface data for the current depth.
 * Requires the rays and intersection buffers.
 */
CPU_ONLY void ExtractSurfaceData(
    unsigned a_NumIntersections,
    WaveFront::AtomicBuffer <WaveFront::IntersectionData>* a_IntersectionData,
    WaveFront::AtomicBuffer <WaveFront::IntersectionRayData>* a_Rays,
    WaveFront::SurfaceData* a_OutPut, 
    SceneDataTableAccessor* a_SceneDataTable);

/*
 * Called each wave after resolving a RayBatch.
 * Shade the intersection points.
 * This does direct, indirect and specular shading.
 * This fills the ShadowRayBatch with potential contributions per pixel and a ray definition.
 * Synchronizes with device at the end of the function.
 */
CPU_ONLY void Shade(const WaveFront::ShadingLaunchParameters& a_ShadingParams);

/*
 * Called at the end of the frame.
 * Apply de-noising, up scaling and post-processing effects.
 */
CPU_ONLY void PostProcess(const WaveFront::PostProcessLaunchParameters& a_PostProcessParams);