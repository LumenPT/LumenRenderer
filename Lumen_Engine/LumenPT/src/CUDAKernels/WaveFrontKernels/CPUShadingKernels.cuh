#pragma once
#include "../../Shaders/CppCommon/CudaDefines.h"
#include "../../Shaders/CppCommon/WaveFrontDataStructs/CudaKernelParamStructs.h"

using namespace WaveFront;

//CPU_GPU void HaltonSequence(
//    int index,
//    int base,
//    float* result);

/*
 * Called at start of frame.
 * Generates the camera rays.
 * Synchronizes with device at the end of the function.
 */
CPU_ONLY void GeneratePrimaryRays(const PrimRayGenLaunchParameters& a_PrimRayGenParams);

/*
 *
 */
CPU_ONLY void GenerateMotionVectors();

/*
 * Called each wave after resolving a RayBatch.
 * Shade the intersection points.
 * This does direct, indirect and specular shading.
 * This fills the ShadowRayBatch with potential contributions per pixel and a ray definition.
 * Synchronizes with device at the end of the function.
 */
CPU_ONLY void Shade(const ShadingLaunchParameters& a_ShadingParams);

/*
 * Called at the end of the frame.
 * Apply de-noising, up scaling and post-processing effects.
 */
CPU_ONLY void PostProcess(const PostProcessLaunchParameters& a_PostProcessParams);