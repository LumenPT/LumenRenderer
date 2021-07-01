#include "CPUShadingKernels.cuh"
#include "GPUShadingKernels.cuh"
#include "../MotionVectors.cuh"
#include "../../Framework/CudaUtilities.h"
#include "../../Shaders/CppCommon/ReSTIRData.h"
#include "../../Framework/ReSTIR.h"
#include <cmath>

using namespace WaveFront;

CPU_ONLY void GeneratePrimaryRays(const PrimRayGenLaunchParameters& a_PrimaryRayGenParams, cudaSurfaceObject_t a_JitterOutput)
{
    const float3 u = a_PrimaryRayGenParams.m_Camera.m_Up;
    const float3 v = a_PrimaryRayGenParams.m_Camera.m_Right;
    const float3 w = a_PrimaryRayGenParams.m_Camera.m_Forward;
    const float3 eye = a_PrimaryRayGenParams.m_Camera.m_Position;
    const uint2 dimensions = uint2{ a_PrimaryRayGenParams.m_Resolution.x, a_PrimaryRayGenParams.m_Resolution.y };
    const int numRays = dimensions.x * dimensions.y;
    const unsigned int frameCount = a_PrimaryRayGenParams.m_FrameCount;

    const int blockSize = 256;
    const int numBlocks = (numRays + blockSize - 1) / blockSize;

    GeneratePrimaryRay << <numBlocks, blockSize >> > (numRays, a_PrimaryRayGenParams.m_PrimaryRays, u, v, w, eye, dimensions, frameCount, a_JitterOutput);
}

CPU_ONLY void GenerateMotionVectors(MotionVectorsGenerationData& a_MotionVectorsData)
{

    const dim3 blockSize{ 16, 16 ,1 };

    const unsigned blockSizeWidth =
        static_cast<unsigned>(std::ceil(static_cast<float>(a_MotionVectorsData.m_RenderResolution.x) / static_cast<float>(blockSize.x)));
    const unsigned blockSizeHeight =
        static_cast<unsigned>(std::ceil(static_cast<float>(a_MotionVectorsData.m_RenderResolution.y) / static_cast<float>(blockSize.y)));

    const dim3 numBlocks{ blockSizeWidth, blockSizeHeight, 1 };

    const sutil::Matrix<4, 4> matrix = a_MotionVectorsData.m_ProjectionMatrix * a_MotionVectorsData.m_PrevViewMatrix;

    sutil::Matrix<4, 4>* matrixDevPtr = { nullptr };
    cudaMalloc(&matrixDevPtr, sizeof(matrix));
    cudaMemcpy(matrixDevPtr, &matrix, sizeof(matrix), cudaMemcpyHostToDevice);

    GenerateMotionVector << <numBlocks, blockSize >> > (
        a_MotionVectorsData.m_MotionVectorBuffer,
        a_MotionVectorsData.m_CurrentSurfaceData,
        a_MotionVectorsData.m_RenderResolution,
        matrixDevPtr);

    cudaFree(matrixDevPtr);

    cudaDeviceSynchronize();
}

CPU_ONLY void ExtractSurfaceData(
    unsigned a_NumIntersections,
    AtomicBuffer<IntersectionData>* a_IntersectionData,
    AtomicBuffer<IntersectionRayData>* a_Rays,
    SurfaceData* a_OutPut,
    cudaSurfaceObject_t a_DepthOutPut,
    uint2 a_Resolution,
    SceneDataTableAccessor* a_SceneDataTable,
    float2 a_MinMaxDepth,
    unsigned int a_CurrentDepth)
{
    const int blockSize = 256;
    const int numBlocks = (a_NumIntersections + blockSize - 1) / blockSize;

    ExtractSurfaceDataGpu << <numBlocks, blockSize >> > (a_NumIntersections, a_IntersectionData, a_Rays, a_OutPut, a_Resolution, a_SceneDataTable);

    cudaDeviceSynchronize();
    if (a_CurrentDepth == 0)
    {

        const dim3 blockSize2d{ 16, 16 ,1 };
        const unsigned blockSizeWidth =
            static_cast<unsigned>(std::ceil(static_cast<float>(a_Resolution.x) / static_cast<float>(blockSize2d.x)));
        const unsigned blockSizeHeight =
            static_cast<unsigned>(std::ceil(static_cast<float>(a_Resolution.y) / static_cast<float>(blockSize2d.y)));

        const dim3 numBlocks2d{ blockSizeWidth, blockSizeHeight, 1 };

        ExtractDepthDataGpu << <numBlocks2d, blockSize2d >> > (a_OutPut, a_DepthOutPut, a_Resolution, a_MinMaxDepth);
    }

}

CPU_ONLY void Shade(const ShadingLaunchParameters& a_ShadingParams)
{

    const dim3 blockSize{ 16, 16 ,1 };

    const unsigned blockSizeWidth =
        static_cast<unsigned>(std::ceil(static_cast<float>(a_ShadingParams.m_ResolutionAndDepth.x) / static_cast<float>(blockSize.x)));
    const unsigned blockSizeHeight =
        static_cast<unsigned>(std::ceil(static_cast<float>(a_ShadingParams.m_ResolutionAndDepth.y) / static_cast<float>(blockSize.y)));

    const dim3 numBlocks{ blockSizeWidth, blockSizeHeight, 1 };

    auto seed = WangHash(a_ShadingParams.m_Seed);
    //TODO
    /*
     * - Implement the below functions.
     * - Access to intersection data, as well as the ray resulting in this shading for chained BRDF scaling.
     */

     /*cudaDeviceSynchronize();
        CHECKLASTCUDAERROR;*/

        //Generate shadow rays for specular highlights.
        /*ShadeSpecular<<<numBlocks, blockSize >>>();*/

        /*cudaDeviceSynchronize();
          CHECKLASTCUDAERROR;*/

          //Ensure that ReSTIR is loaded so that the CDF can be extracted.
    assert(a_ShadingParams.m_ReSTIR != nullptr);

    /*
     * First depth is somewhat of a special case.
     */
    if (a_ShadingParams.m_CurrentDepth == 0)
    {
        /*
         * First visualize all lights directly hit by the camera rays.
         */
        ResolveDirectLightHits << <numBlocks, blockSize >> > (
            a_ShadingParams.m_CurrentSurfaceData,
            uint2{ a_ShadingParams.m_ResolutionAndDepth.x, a_ShadingParams.m_ResolutionAndDepth.y },
            a_ShadingParams.m_OutputChannels[static_cast<unsigned>(LightChannel::DIRECT)]
            );

        /*
         * Run ReSTIR to find the best direct light candidates.
         */
        a_ShadingParams.m_ReSTIR->Run(
            a_ShadingParams.m_CurrentSurfaceData,
            a_ShadingParams.m_TemporalSurfaceData,
            a_ShadingParams.m_MotionVectorBuffer,
            a_ShadingParams.m_OptixWrapper,
            a_ShadingParams.m_OptixSceneHandle,
            a_ShadingParams.m_Seed,
            a_ShadingParams.m_LightDataBuffer,
            a_ShadingParams.m_OutputChannels,
            *a_ShadingParams.m_FrameStats,
            true);
    }
    else
    {
        //This was from when ReSTIR was optional. Now it always runs so honestly no need to check.
        ////Generate the CDF if ReSTIR is disabled.
        //if(a_ShadingParams.m_CurrentDepth == 0 && !useRestir)
        //{
        //    a_ShadingParams.m_ReSTIR->BuildCDF(a_ShadingParams.m_TriangleLights);
        //}

        CDF* CDFPtr = a_ShadingParams.m_ReSTIR->GetCdfGpuPointer();

        //Generate shadow rays for direct lights.

        ShadeDirect << <numBlocks, blockSize >> > (
            a_ShadingParams.m_ResolutionAndDepth,
            a_ShadingParams.m_CurrentSurfaceData,
            a_ShadingParams.m_CurrentVolumetricData,
            a_ShadingParams.m_LightDataBuffer->GetDevicePtr<AtomicBuffer<TriangleLight>>(),
            a_ShadingParams.m_Seed,
            CDFPtr,
            a_ShadingParams.m_SolidShadowRayBuffer,
            a_ShadingParams.m_VolumetricShadowRayBuffer,
            a_ShadingParams.m_OutputChannels[static_cast<unsigned>(LightChannel::VOLUMETRIC)]);
    }

    cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;

    //Update the seed.
    seed = WangHash(a_ShadingParams.m_Seed);

    //Generate secondary rays only when there's a wave after this.
    if (a_ShadingParams.m_CurrentDepth < a_ShadingParams.m_ResolutionAndDepth.z - 1)
    {

        ShadeIndirect << <numBlocks, blockSize >> > (
            a_ShadingParams.m_ResolutionAndDepth,
            a_ShadingParams.m_CurrentSurfaceData,
            a_ShadingParams.m_RayBuffer,
            seed);

    }

    cudaDeviceSynchronize();
}

CPU_ONLY void PostProcess(const PostProcessLaunchParameters& a_PostProcessParams)
{

    //The amount of pixels and threads/blocks needed to apply effects.
    const dim3 blockSize{ 32, 32 ,1 };

    const unsigned blockSizeWidth =
        static_cast<unsigned>(std::ceil(static_cast<float>(a_PostProcessParams.m_RenderResolution.x) / static_cast<float>(blockSize.x)));
    const unsigned blockSizeHeight =
        static_cast<unsigned>(std::ceil(static_cast<float>(a_PostProcessParams.m_RenderResolution.y) / static_cast<float>(blockSize.y)));

    const dim3 numBlocks{ blockSizeWidth, blockSizeHeight, 1 };

    //TODO before merging.
    //Denoise<<<numBlocks, blockSize >>>();

    /*cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;*/

    MergeOutputChannels << <numBlocks, blockSize >> > (
        a_PostProcessParams.m_RenderResolution,
        a_PostProcessParams.m_PixelBufferMultiChannel,
        a_PostProcessParams.m_PixelBufferSingleChannel,
        a_PostProcessParams.m_BlendOutput,
        a_PostProcessParams.m_BlendCount
        );

    /*cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;*/

    //TODO steal hidden Nvidia technology by breaking into their buildings
    //TODO copy merged results into image output after having run DLSS.
    DLSS << <numBlocks, blockSize >> > ();

    const unsigned blockSizeWidthUpscaled =
        static_cast<unsigned>(std::ceil(static_cast<float>(a_PostProcessParams.m_OutputResolution.x) / static_cast<float>(blockSize.x)));
    const unsigned blockSizeHeightUpscaled =
        static_cast<unsigned>(std::ceil(static_cast<float>(a_PostProcessParams.m_OutputResolution.y) / static_cast<float>(blockSize.y)));

    const dim3 numBlocksUpscaled{ blockSizeWidthUpscaled, blockSizeHeightUpscaled, 1 };

    /*cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;*/

    //TODO
    //PostProcessingEffects << <numBlocksUpscaled, blockSizeUpscaled >> > ();

    /*cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;*/

    //TODO This is temporary till the post-processing is  in place. Let the last stage copy it directly to the output buffer.
    WriteToOutput << <numBlocksUpscaled, blockSize >> > (
        a_PostProcessParams.m_OutputResolution,
        a_PostProcessParams.m_PixelBufferSingleChannel,
        a_PostProcessParams.m_FinalOutput
        );
}

CPU_ONLY void MergeOutput(const PostProcessLaunchParameters& a_PostProcessParams)
{
    //The amount of pixels and threads/blocks needed to apply effects.
    const dim3 blockSize{ 32, 32 ,1 };

    const unsigned blockSizeWidth =
        static_cast<unsigned>(std::ceil(static_cast<float>(a_PostProcessParams.m_RenderResolution.x) / static_cast<float>(blockSize.x)));
    const unsigned blockSizeHeight =
        static_cast<unsigned>(std::ceil(static_cast<float>(a_PostProcessParams.m_RenderResolution.y) / static_cast<float>(blockSize.y)));

    const dim3 numBlocks{ blockSizeWidth, blockSizeHeight, 1 };

    MergeOutputChannels << <numBlocks, blockSize >> > (
        a_PostProcessParams.m_RenderResolution,
        a_PostProcessParams.m_PixelBufferMultiChannel,
        a_PostProcessParams.m_PixelBufferSingleChannel,
        a_PostProcessParams.m_BlendOutput,
        a_PostProcessParams.m_BlendCount
        );
}

CPU_ONLY void WriteToOutput(const WriteOutputParams& a_WriteOutputParams)
{
    //The amount of pixels and threads/blocks needed to apply effects.
    const dim3 blockSize{ 32, 32 ,1 };

    const unsigned blockSizeWidthUpscaled =
        static_cast<unsigned>(std::ceil(static_cast<float>(a_WriteOutputParams.m_OutputResolution.x) / static_cast<float>(blockSize.x)));
    const unsigned blockSizeHeightUpscaled =
        static_cast<unsigned>(std::ceil(static_cast<float>(a_WriteOutputParams.m_OutputResolution.y) / static_cast<float>(blockSize.y)));

    const dim3 numBlocksUpscaled{ blockSizeWidthUpscaled, blockSizeHeightUpscaled, 1 };

    //TODO This is temporary till the post-processing is  in place. Let the last stage copy it directly to the output buffer.
    WriteToOutput << <numBlocksUpscaled, blockSize >> > (
        a_WriteOutputParams.m_OutputResolution,
        a_WriteOutputParams.m_PixelBufferSingleChannel,
        a_WriteOutputParams.m_FinalOutput
        );
}

CPU_ONLY void PrepareOptixDenoising(WaveFront::OptixDenoiserLaunchParameters& a_LaunchParams)
{
    //The amount of pixels and threads/blocks needed to apply effects.
    const dim3 blockSize{ 32, 32 ,1 };

    const unsigned blockSizeWidth =
        static_cast<unsigned>(std::ceil(static_cast<float>(a_LaunchParams.m_RenderResolution.x) / static_cast<float>(blockSize.x)));
    const unsigned blockSizeHeight =
        static_cast<unsigned>(std::ceil(static_cast<float>(a_LaunchParams.m_RenderResolution.y) / static_cast<float>(blockSize.y)));

    const dim3 numBlocks{ blockSizeWidth, blockSizeHeight, 1 };

    PrepareOptixDenoisingGPU << <numBlocks, blockSize >> > (
        a_LaunchParams.m_RenderResolution,
        a_LaunchParams.m_CurrentSurfaceData,
        a_LaunchParams.m_PixelBufferSingleChannel,
        a_LaunchParams.m_IntermediaryInput,
        a_LaunchParams.m_AlbedoInput,
        a_LaunchParams.m_NormalInput,
        a_LaunchParams.m_FlowInput,
        a_LaunchParams.m_IntermediaryOutput
        );
}

CPU_ONLY void FinishOptixDenoising(WaveFront::OptixDenoiserLaunchParameters& a_LaunchParams)
{
    //The amount of pixels and threads/blocks needed to apply effects.
    const dim3 blockSize{ 32, 32 ,1 };

    const unsigned blockSizeWidth =
        static_cast<unsigned>(std::ceil(static_cast<float>(a_LaunchParams.m_RenderResolution.x) / static_cast<float>(blockSize.x)));
    const unsigned blockSizeHeight =
        static_cast<unsigned>(std::ceil(static_cast<float>(a_LaunchParams.m_RenderResolution.y) / static_cast<float>(blockSize.y)));

    const dim3 numBlocks{ blockSizeWidth, blockSizeHeight, 1 };

    FinishOptixDenoisingGPU << <numBlocks, blockSize >> > (
        a_LaunchParams.m_RenderResolution,
        a_LaunchParams.m_PixelBufferSingleChannel,
        a_LaunchParams.m_IntermediaryInput,
        a_LaunchParams.m_IntermediaryOutput,
        a_LaunchParams.m_BlendOutput,
        a_LaunchParams.m_UseBlendOutput,
        a_LaunchParams.m_BlendCount
        );
}