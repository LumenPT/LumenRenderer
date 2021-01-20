#pragma once

#include <cuda_runtime.h>
#include "../Shaders/CppCommon/WaveFrontDataStructs.h"

//Some defines to make the functions less scary and more readable

#define GPU_ONLY __device__ __forceinline__ //Runs on GPU only, available on GPU only.
#define CPU_GPU __global__ __forceinline__ //Runs on GPU, available on GPU and CPU.
#define CPU_ONLY __host__

using namespace WaveFront;

/*
 * Shade the intersection points.
 * This does direct, indirect and specular shading.
 * This fills the shadow ray buffer with potential contributions per pixel.
 */
CPU_ONLY void Shade(const ShadingLaunchParameters& a_ShadingParams);

/*
 * Apply de-noising, up scaling and post-processing effects.
 */
CPU_ONLY void PostProcess(const PostProcessLaunchParameters& a_PostProcessParams);


//The below functions are only called internally from the GPU_ONLY within the above defined functions.

//Called in setup.
CPU_ONLY void GenerateRays(const SetupLaunchParameters& a_SetupParams);
CPU_ONLY void GenerateMotionVectors();

//Called during shading
CPU_GPU void ShadeDirect(const uint3& a_ResolutionAndDepth, const RayBatch* const a_CurrentRays, const IntersectionBuffer* const a_Intersections, ShadowRayBatch* const a_ShadowRays, const LightBuffer* const a_Lights);
CPU_GPU void ShadeSpecular();
CPU_GPU void ShadeIndirect(const uint3& a_ResolutionAndDepth, const IntersectionBuffer* const a_Intersections, const RayBatch* const a_PrimaryRays, RayBatch* a_Output);

//Called during post-processing.
CPU_GPU void Denoise();
CPU_GPU void MergeLightChannels(int a_NumPixels, const uint2& a_Dimensions, const PixelBuffer* const a_Input, PixelBuffer* const a_Output);
CPU_GPU void DLSS();
CPU_GPU void PostProcessingEffects();

//Temporary step till post-processing is in place.
CPU_GPU void WriteToOutput(int a_NumPixels, const uint2& a_Dimensions, PixelBuffer* const a_Input, uchar4* a_Output);


//Generate some rays based on the thread index.
CPU_GPU void GenerateRay(int a_NumRays, RayBatch* const a_Buffer, const float3& a_U, const float3& a_V, const float3& a_W, const float3& a_Eye, const int2& a_Dimensions);