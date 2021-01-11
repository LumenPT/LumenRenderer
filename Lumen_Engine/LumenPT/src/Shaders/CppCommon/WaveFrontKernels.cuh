#pragma once

#include <cuda_runtime.h>
#include <cinttypes>
#include "./WaveFrontDataStructs.h"

//Some defines to make the functions less scary and more readable
#define GPU __device__ __forceinline__ //Runs on GPU only, available on GPU only.
#define CPU __global__ __forceinline__ //Runs on GPU, available on GPU and CPU.
#define CPU_ONLY __host__ __forceinline__

using namespace WaveFront;

/*
 * Called once before every frame.
 * Generates primary rays from the camera etc.
 * Rays are stored in the ray batch.
 */
CPU void PreRenderSetup(const SetupLaunchParameters& a_SetupParams);

/*
 * Call Optix to intersect all rays in the ray batch.
 * Stores intersection data in the intersection data buffer.
 */
CPU_ONLY void IntersectRays();

/*
 * Shade the intersection points.
 * This does direct, indirect and specular shading.
 * This fills the shadow ray buffer with potential contributions per pixel.
 */
CPU void Shade(const ShadingLaunchParameters& a_ShadingParams);

/*
 * This resolves all shadow rays in parallel, and adds the light contribution
 * of each light channel to the output pixel buffer for each un-occluded ray.
 */
CPU_ONLY void ResolveShadowRays();

/*
 * Apply de-noising, up scaling and post-processing effects.
 */
CPU void PostProcess(const PostProcessLaunchParameters& a_PostProcessParams);


//The below functions are only called internally from the GPU within the above defined functions.

//Called in setup.
GPU void GenerateRays();
GPU void GenerateMotionVectors();

//Called during shading
GPU void ShadeDirect();
GPU void ShadeSpecular();
GPU void ShadeIndirect();

//Called during post-processing.
GPU void Denoise();
GPU void MergeLightChannels();
GPU void DLSS();
GPU void PostProcessingEffects();