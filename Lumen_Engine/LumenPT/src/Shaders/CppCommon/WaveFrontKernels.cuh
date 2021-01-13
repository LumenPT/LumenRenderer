#pragma once

#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <cinttypes>
#include "./WaveFrontDataStructs.h"

//Some defines to make the functions less scary and more readable
#define GPU __device__ __forceinline__ //Runs on GPU only, available on GPU only.
#define CPU __global__ __forceinline__ //Runs on GPU, available on GPU and CPU.
#define CPU_ONLY __host__ __forceinline__

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


//The below functions are only called internally from the GPU within the above defined functions.

//Called in setup.
CPU_ONLY void GenerateRays(const SetupLaunchParameters& a_SetupParams);
CPU_ONLY void GenerateMotionVectors();

//Called during shading
CPU void ShadeDirect();
CPU void ShadeSpecular();
CPU void ShadeIndirect();

//Called during post-processing.
GPU void Denoise();
CPU void MergeLightChannels(int a_NumPixels, const uint2& a_Dimensions, PixelBuffer* a_Input, float3* a_Output);
GPU void DLSS();
GPU void PostProcessingEffects();


//Generate some rays based on the thread index.
CPU void GenerateRay(int a_NumRays, RayData* a_Buffer, const float3& a_U, const float3& a_V, const float3& a_W, const float3& a_Eye, const int2& a_Dimensions);