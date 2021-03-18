#include "CPUShadingKernels.cuh"
#include "GPUShadingKernels.cuh"
#include "../../Framework/CudaUtilities.h"

using namespace WaveFront;

//CPU_GPU void HaltonSequence(
//    int index,
//    int base,
//    float* result)
//{
//    ++index;
//
//    float f = 1.f;
//    float r = 0.f;
//
//    while (index > 0)
//    {
//        f = f / base;
//        r = r + f * (index % base);
//        index = index / base;
//    }
//
//    *result = r;
//}

CPU_ONLY void GeneratePrimaryRays(const PrimRayGenLaunchParameters& a_PrimaryRayGenParams)
{
    const float3 u = a_PrimaryRayGenParams.m_Camera.m_Up;
    const float3 v = a_PrimaryRayGenParams.m_Camera.m_Right;
    const float3 w = a_PrimaryRayGenParams.m_Camera.m_Forward;
    const float3 eye = a_PrimaryRayGenParams.m_Camera.m_Position;
    const int2 dimensions = make_int2(a_PrimaryRayGenParams.m_Resolution.x, a_PrimaryRayGenParams.m_Resolution.y);
    const int numRays = dimensions.x * dimensions.y;
    const unsigned int frameCount = a_PrimaryRayGenParams.m_FrameCount;

    const int blockSize = 256;
    const int numBlocks = (numRays + blockSize - 1) / blockSize;

    GeneratePrimaryRay<<<numBlocks, blockSize>>>(numRays, a_PrimaryRayGenParams.m_PrimaryRays, u, v, w, eye, dimensions, frameCount);
}

CPU_ONLY void GenerateMotionVectors()
{
}

CPU_ONLY void ExtractSurfaceData(
    unsigned a_NumIntersections, 
    AtomicBuffer < IntersectionData>* a_IntersectionData, 
    AtomicBuffer < IntersectionRayData>* a_Rays, 
    SurfaceData* a_OutPut, 
    SceneDataTableAccessor* a_SceneDataTable)
{
    const int blockSize = 256;
    const int numBlocks = (a_NumIntersections + blockSize - 1) / blockSize;

    ExtractSurfaceDataGpu<<<numBlocks, blockSize>>>(a_NumIntersections, a_IntersectionData, a_Rays, a_OutPut, a_SceneDataTable);
}

CPU_ONLY void Shade(const ShadingLaunchParameters& a_ShadingParams)
{
    //TODO
    /*
     * - Implement the below functions.
     * - Access to intersection data, as well as the ray resulting in this shading for chained BRDF scaling.
     */

    const int numPixels = a_ShadingParams.m_ResolutionAndDepth.x * a_ShadingParams.m_ResolutionAndDepth.y;
    const int blockSize = 256;
    const int numBlocks = (numPixels + blockSize - 1) / blockSize;

    //Generate secondary rays.

    /*ShadeIndirect<<<numBlocks,blockSize>>>(
       a_ShadingParams.m_ResolutionAndDepth,
       a_ShadingParams.m_CurrentRays,
       a_ShadingParams.m_CurrentIntersections,
       a_ShadingParams.m_SecondaryRays);*/

    /*cudaDeviceSynchronize();
       CHECKLASTCUDAERROR;*/

    //Generate shadow rays for specular highlights.
    /*ShadeSpecular<<<numBlocks, blockSize >>>();*/

    /*cudaDeviceSynchronize();
      CHECKLASTCUDAERROR;*/

    //Generate shadow rays for direct lights.
    ShadeDirect<<<numBlocks, blockSize>>>(
        a_ShadingParams.m_ResolutionAndDepth, 
        a_ShadingParams.m_TemporalSurfaceData, 
        a_ShadingParams.m_CurrentSurfaceData, 
        a_ShadingParams.m_ShadowRays, 
        a_ShadingParams.m_TriangleLights, 
        a_ShadingParams.m_NumLights, 
        a_ShadingParams.m_CDF);

    /*DEBUGShadePrimIntersections<<<numBlocks, blockSize>>>(
        a_ShadingParams.m_ResolutionAndDepth,
        a_ShadingParams.m_CurrentSurfaceData,
        a_ShadingParams.m_Output);*/
}

CPU_ONLY void PostProcess(const PostProcessLaunchParameters& a_PostProcessParams)
{
    //TODO
    /*
     * Not needed now. Can be implemented later.
     * For now just merge the final light contributions to get the final pixel color.
     */

    //The amount of pixels and threads/blocks needed to apply effects.
    const int numPixels = a_PostProcessParams.m_RenderResolution.x * a_PostProcessParams.m_RenderResolution.y;
    const int blockSize = 256;
    const int numBlocks = (numPixels + blockSize - 1) / blockSize;

    //TODO before merging.
    //Denoise<<<numBlocks, blockSize >>>();

    /*cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;*/

    MergeOutputChannels << <numBlocks, blockSize >> > (
        a_PostProcessParams.m_RenderResolution,
        a_PostProcessParams.m_WavefrontOutput,
        a_PostProcessParams.m_ProcessedOutput);

    /*cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;*/

    //TODO steal hidden Nvidia technology by breaking into their buildings
    //TODO copy merged results into image output after having run DLSS.
    DLSS << <numBlocks, blockSize >> > ();

    const int numPixelsUpscaled = a_PostProcessParams.m_OutputResolution.x * a_PostProcessParams.m_OutputResolution.y;
    const int blockSizeUpscaled = 256;
    const int numBlocksUpscaled = (numPixelsUpscaled + blockSizeUpscaled - 1) / blockSizeUpscaled;

    /*cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;*/

    //TODO
    //PostProcessingEffects << <numBlocksUpscaled, blockSizeUpscaled >> > ();

    /*cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;*/

    //TODO This is temporary till the post-processing is  in place. Let the last stage copy it directly to the output buffer.
    WriteToOutput << <numBlocksUpscaled, blockSizeUpscaled >> > (
        a_PostProcessParams.m_OutputResolution,
        a_PostProcessParams.m_ProcessedOutput,
        a_PostProcessParams.m_FinalOutput);
}