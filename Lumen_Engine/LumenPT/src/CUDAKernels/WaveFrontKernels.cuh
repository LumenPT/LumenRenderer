#pragma once

#include <cuda_runtime.h>
#include "../Shaders/CppCommon/WaveFrontDataStructs.h"

//Some defines to make the functions less scary and more readable

#define GPU_ONLY __device__ __forceinline__ //Runs on GPU only, available on GPU only.
#define CPU_GPU __global__ //Runs on GPU, available on GPU and CPU.
#define CPU_ONLY __host__

using namespace WaveFront;






/*
 * Called at start of frame.
 * Generates the camera rays.
 * Synchronizes with device at the end of the function.
 */
CPU_ONLY void GenerateRays(const SetupLaunchParameters& a_SetupParams);

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



//The below functions are only called internally from the GPU_ONLY within the above defined functions.

//Generate some rays based on the thread index.
CPU_GPU void GenerateRay(
    int a_NumRays, 
    RayBatch* const a_Buffer, 
    float3 a_U, 
    float3 a_V, 
    float3 a_W, 
    float3 a_Eye, 
    int2 a_Dimensions);

//Called during shading

/*
 * Called at the start of the Shade function with a number of thread-blocks.
 * Calculates direct shading as potential light contribution.
 * Defines a shadow ray that is used to validate the potential light contribution.
 */
CPU_GPU void ShadeDirect(
    const uint3 a_ResolutionAndDepth, 
    const RayBatch* const a_CurrentRays, 
    const IntersectionBuffer* const a_CurrentIntersections, 
    ShadowRayBatch* const a_ShadowRays, 
    const LightBuffer* const a_Lights, 
    CDF* const a_CDF /*const CDF* a_CDF*/);

/*
 *
 */
CPU_GPU void ShadeSpecular();

/*
 *
 */
CPU_GPU void ShadeIndirect(
    const uint3 a_ResolutionAndDepth, 
    const IntersectionBuffer* const a_Intersections, 
    const RayBatch* const a_PrimaryRays, 
    RayBatch* const a_Output);

//Called during post-processing.

/*
 *
 */
CPU_GPU void Denoise();

/*
 *
 */
CPU_GPU void MergeLightChannels(
    const uint2 a_Resolution, 
    const ResultBuffer* const a_Input, 
    PixelBuffer* const a_Output);

/*
 *
 */
CPU_GPU void DLSS();

/*
 *
 */
CPU_GPU void PostProcessingEffects();

//Temporary step till post-processing is in place.
CPU_GPU void WriteToOutput( 
    const uint2 a_Resolution, 
    const PixelBuffer* const a_Input, 
    uchar4* a_Output);

GPU_ONLY inline unsigned int WangHash(unsigned int a_S)
{
    a_S = (a_S ^ 61) ^ (a_S >> 16), a_S *= 9, a_S = a_S ^ (a_S >> 4), a_S *= 0x27d4eb2d, a_S = a_S ^ (a_S >> 15); return a_S;
}

GPU_ONLY inline unsigned int RandomInt(unsigned int& a_S)
{
    a_S ^= a_S << 13, a_S ^= a_S >> 17, a_S ^= a_S << 5; return a_S;
};

GPU_ONLY inline float RandomFloat(unsigned int& a_S)
{
    return RandomInt(a_S) * 2.3283064365387e-10f;
}