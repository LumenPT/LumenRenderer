#pragma once
#include "../../Shaders/CppCommon/CudaDefines.h"
#include "../../Shaders/CppCommon/WaveFrontDataStructs.h"
#include "../../Shaders/CppCommon/ArrayParameter.h"

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
    uint2 a_Dimensions,
    unsigned int a_FrameCount,
    cudaSurfaceObject_t a_JitterOutput);

CPU_ON_GPU void ExtractSurfaceDataGpu(unsigned a_NumIntersections,
    AtomicBuffer<IntersectionData>* a_IntersectionData,
    AtomicBuffer<IntersectionRayData>* a_Rays,
    SurfaceData* a_OutPut,
    uint2 a_Resolution,
    SceneDataTableAccessor* a_SceneDataTable);

    
CPU_ON_GPU void ExtractDepthDataGpu(
    const SurfaceData* a_SurfaceData,
    cudaSurfaceObject_t a_DepthOutPut,
    uint2 a_Resolution,
    float2 a_MinMaxDistance);

CPU_ON_GPU void ExtractNRD_DLSSdataGpu(
    const SurfaceData* a_SurfaceData,
    cudaSurfaceObject_t a_DepthOutPut,
    cudaSurfaceObject_t a_NormalRoughnessOutput,
    uint2 a_Resolution,
    float2 a_MinMaxDistance);

//Called during shading

/*
 * Called at the start of the Shade function with a number of thread-blocks.
 * Calculates direct shading as potential light contribution.
 * Defines a shadow ray that is used to validate the potential light contribution.
 */
CPU_ON_GPU void ShadeDirect(
    const uint3 a_ResolutionAndDepth,
    const SurfaceData* a_SurfaceDataBuffer,
    const VolumetricData* a_VolumetricDataBuffer,
    const AtomicBuffer<TriangleLight>* const a_Lights,
    const unsigned a_Seed,
    const CDF* const a_CDF,
    AtomicBuffer<ShadowRayData>* const a_ShadowRays,
    AtomicBuffer<ShadowRayData>* const a_VolumetricShadowRays,
    cudaSurfaceObject_t a_VolumetricOutput		//TODO: remove a_Output
    );

/*
 * When a light is hit at depth 0, it needs to be visualized on the screen.
 * This kernel does that.
 */
CPU_ON_GPU void ResolveDirectLightHits(
    const SurfaceData* a_SurfaceDataBuffer,
    const uint2 a_Resolution,
    cudaSurfaceObject_t a_OutputChannels
);

/*
 *
 */
CPU_ON_GPU void ShadeSpecular();

/*
 *
 */
CPU_ON_GPU void ShadeIndirect(
    const uint3 a_ResolutionAndDepth,
    const SurfaceData* a_SurfaceDataBuffer,
    AtomicBuffer<IntersectionRayData>* a_IntersectionRays,
    const unsigned a_Seed
);



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
    const ArrayParameter<cudaSurfaceObject_t, static_cast<unsigned>(LightChannel::NUM_CHANNELS)> a_Input,
    const cudaSurfaceObject_t a_Output,
    const bool a_BlendOutput,
    const unsigned a_BlendCount
);

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
    const cudaSurfaceObject_t a_Input,
    uchar4* a_Output
);

CPU_ON_GPU void PrepareOptixDenoisingGPU(
    const uint2 a_RenderResolution,
    const SurfaceData* a_CurrentSurfaceData,
    const cudaSurfaceObject_t a_PixelBufferSingleChannel,
    float3* a_IntermediaryInput,
    float3* a_AlbedoInput,
    float3* a_NormalInput,
    float2* a_FlowInput,
    float3* a_IntermediaryOutput);

CPU_ON_GPU void FinishOptixDenoisingGPU(
    const uint2 a_RenderResolution,
    const cudaSurfaceObject_t a_PixelBufferSingleChannel,
    float3* a_IntermediaryInput,
    float3* a_IntermediaryOutput,
    float3* a_BlendOutput,
    bool a_UseBlendOutput,
    unsigned int a_BlendCount);