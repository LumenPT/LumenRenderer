#pragma once
#include "../../Shaders/CppCommon/CudaDefines.h"
#include "../../Shaders/CppCommon/WaveFrontDataStructs/CudaKernelParamStructs.h"

class SceneDataTableAccessor;

struct ExtractSurfaceDataParams
{

    unsigned m_NumIntersections;
    WaveFront::AtomicBuffer <WaveFront::IntersectionData>* m_IntersectionData;
    WaveFront::AtomicBuffer <WaveFront::IntersectionRayData>* m_Rays;
    SceneDataTableAccessor* m_SceneDataTable;
    WaveFront::SurfaceData* m_SurfaceDataOutPut;
    cudaSurfaceObject_t m_DepthOutPut;
    cudaSurfaceObject_t m_NormalRoughnessOutput;
    uint2 m_Resolution;
    float2 m_MinMaxDepth;
    unsigned int m_CurrentDepth;

};


/*
 * Called at start of frame.
 * Generates the camera rays.
 * Synchronizes with device at the end of the function.
 */
CPU_ONLY void GeneratePrimaryRays(const WaveFront::PrimRayGenLaunchParameters& a_PrimRayGenParams, cudaSurfaceObject_t a_JitterOutput);

/*
 *
 */
CPU_ONLY void GenerateMotionVectors(WaveFront::MotionVectorsGenerationData& a_MotionVectorsData);

/*
 * Extract the surface data for the current depth.
 * Requires the rays and intersection buffers.
 */
CPU_ONLY void ExtractSurfaceData(const ExtractSurfaceDataParams& a_ExtractionParams);

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

CPU_ONLY void MergeOutput(const WaveFront::PostProcessLaunchParameters& a_PostProcessParams);

CPU_ONLY void WriteToOutput(const WaveFront::PostProcessLaunchParameters& a_PostProcessParams);

CPU_ONLY void PrepareOptixDenoising(WaveFront::OptixDenoiserLaunchParameters& a_LaunchParams);

CPU_ONLY void FinishOptixDenoising(WaveFront::OptixDenoiserLaunchParameters& a_LaunchParams);