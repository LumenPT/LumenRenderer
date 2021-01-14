#include "CppCommon/WaveFrontKernels.cuh"

#include "../../vendor/Include/Cuda/cuda/helpers.h"
#include "CppCommon/RenderingUtility.h"

#include "device_launch_parameters.h"
#include "../../vendor/Include/sutil/vec_math.h"

using namespace WaveFront;

CPU_ONLY void GenerateRays(const SetupLaunchParameters& a_SetupParams)
{
    const float3 u = a_SetupParams.m_Camera.m_Up;
    const float3 v = a_SetupParams.m_Camera.m_Right;
    const float3 w = a_SetupParams.m_Camera.m_Forward;
    const float3 eye = a_SetupParams.m_Camera.m_Position;
    const int2 dimensions = make_int2(a_SetupParams.m_Resolution.x, a_SetupParams.m_Resolution.y);
    const int numRays = dimensions.x * dimensions.y;

    const int blockSize = 256;
    const int numBlocks = (numRays + blockSize - 1) / blockSize;

    GenerateRay <<<numBlocks, blockSize >>> (numRays, a_SetupParams.m_PrimaryRays, u, v, w, eye, dimensions);
}



CPU_ONLY void Shade(const ShadingLaunchParameters& a_ShadingParams)
{
    //TODO
    /*
     * - Implement the below functions.
     * - Access to intersection data, as well as the ray resulting in this shading for chained BRDF scaling.
     */

     //Generate secondary rays.
    ShadeIndirect<<<1,1>>>(a_ShadingParams.m_ResolutionAndDepth, a_ShadingParams.m_Intersections, a_ShadingParams.m_SecondaryRays); 
     //Generate shadow rays for specular highlights.
    ShadeSpecular<<<1,1>>>();
    //Generate shadow rays for direct lights.
    ShadeDirect<<<1,1>>>(a_ShadingParams.m_ResolutionAndDepth, a_ShadingParams.m_Intersections, a_ShadingParams.m_ShadowRaysBatch, a_ShadingParams.m_LightBuffer);   
}


CPU_ONLY void PostProcess(const PostProcessLaunchParameters& a_PostProcessParams)
{
    //TODO
    /*
     * Not needed now. Can be implemented later.
     * For now just merge the final light contributions to get the final pixel color.
     */

     //The amount of pixels and threads/blocks needed to apply effects.
    const int numPixels = a_PostProcessParams.m_Resolution.x * a_PostProcessParams.m_Resolution.y;
    const int blockSize = 256;
    const int numBlocks = (numPixels + blockSize - 1) / blockSize;

    //TODO before merging.
    Denoise();

    MergeLightChannels <<<numBlocks, blockSize >>> (
        numPixels, 
        a_PostProcessParams.m_Resolution, 
        a_PostProcessParams.m_WavefrontOutput->m_PixelOutput,
        a_PostProcessParams.m_MergedResults);

    //TODO steal hidden Nvidia technology by breaking into their buildings
    //TODO copy merged results into image output after having run DLSS.
    DLSS();

    //TODO
    PostProcessingEffects();



    //TODO This is temporary till the post-processing is  in place. Let the last stage copy it directly to the output buffer.
    WriteToOutput<<<numBlocks, blockSize>>>(numPixels, a_PostProcessParams.m_Resolution, a_PostProcessParams.m_MergedResults, a_PostProcessParams.m_ImageOutput);
}



CPU_ONLY void GenerateMotionVectors()
{
}

CPU_GPU void ShadeDirect(
    const uint3& a_ResolutionAndDepth, 
    const IntersectionBuffer* const a_Intersections, 
    ShadowRayBatch* const a_ShadowRays,
    const LightBuffer* const a_Lights)
{
    // need access to light & mesh
        // I get triangleID and meshID, so need to use that to look up actual data based on those IDs
        // 

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const int numRays =a_ResolutionAndDepth.x * a_ResolutionAndDepth.y;

    float3 origin = { 0.0f,0.0f,0.0f };

    /*
     * a_ShadingParams->m_Intersections.m_Intersections[i]  //      intersection data
     *              find intersecting triangle properties through MeshID & TriangleID
     */

    float3 direction = { 0.0f,0.0f,0.0f };
    float3 potRadiance = { 0.0f,0.0f,0.0f };

    for (unsigned int i = index; i < numRays; i += stride)
    {
        //temp and probably shit, looping through all lights in buffer (rather than those nearby)
        for (unsigned int j = 0; i < a_Lights->m_Size; j++)
        {
            direction = normalize(a_Lights->m_Lights[j].m_Position);// - intersectionPosition;

        }

        //write shadow ray direction etc. to shadow ray batch   // TEMP
        ShadowRayData shadowRay(
            origin,
            direction,
            1000.f,
            potRadiance,    //not sure what this represents
            ResultBuffer::OutputChannel::DIRECT
        );

        a_ShadowRays->SetShadowRay(shadowRay, a_ResolutionAndDepth.z, i, 0);
    }

    // look at generaterays function

    // just need a pointer to start of buffer to start processing data

    // need to get material and properties from intersection

    // call BRDF functions

    // buffers may require regular device pointers(?)

    // hold array of buffer handles, so you can look up the buffers that belong to a certain mesh
        // then inside those buffers you look up the triangleID to get the properties of the actual triangle

}

CPU_GPU void ShadeSpecular()
{
}

CPU_GPU void ShadeIndirect(const uint3& a_ResolutionAndDepth, const IntersectionBuffer* const a_Intersections, const RayBatch* const a_PrimaryRays, RayBatch* a_Output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const int numPixels = a_ResolutionAndDepth.x * a_ResolutionAndDepth.y;

    for (int i = index; i < numPixels; i += stride)
    {
        //Convert the index into the screen dimensions.
        const int screenY = i / a_ResolutionAndDepth.x;
        const int screenX = i - (screenY * a_ResolutionAndDepth.x);

        auto& intersection = a_Intersections[i];

        //TODO russian roulette to terminate path (multiply weight with russian roulette outcome
        float russianRouletteWeight = 1.f;

        //TODO extract surface normal from intersection data.
        float3 normal = make_float3(1.f, 0.f, 0.f);

        //TODO generate random direction
        float3 dir = make_float3(1.f, 0.f, 0.f);

        //TODO get position from intersection data.
        float3 pos = make_float3(0.f, 0.f, 0.f);

        //TODO calculate BRDF to see how much light is transported.
        float3 brdf = make_float3(1.f, 1.f, 1.f); //Do this after direct shading because the material is already looked up there. Use the BRDF.
        float3 totalLightTransport = russianRouletteWeight * a_PrimaryRays->m_Rays[i].m_Contribution * brdf;  //Total amount of light that will end up in the camera through this path.

        //Add to the output buffer.
        a_Output->m_Rays[i] = RayData{ pos, dir, totalLightTransport };
    }
}

GPU_ONLY void Denoise()
{
}

CPU_GPU void MergeLightChannels(
    int a_NumPixels, 
    const uint2& a_Dimensions, 
    const PixelBuffer* const a_Input, 
    PixelBuffer* const a_Output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < a_NumPixels; i += stride)
    {
        //Convert the index into the screen dimensions.
        const int screenY = i / a_Dimensions.x;
        const int screenX = i - (screenY * a_Dimensions.x);

        //Mix the results.
        const float3& direct = a_Input->GetPixel(i, static_cast<unsigned>(ResultBuffer::OutputChannel::DIRECT));
        const float3& indirect = a_Input->GetPixel(i, static_cast<unsigned>(ResultBuffer::OutputChannel::INDIRECT));
        const float3& specular = a_Input->GetPixel(i, static_cast<unsigned>(ResultBuffer::OutputChannel::SPECULAR));
        //a_Output[i] = data[0] + data[1] + data[2]; this isnt correct ? take a float 3 and add each of its members to each other ?

        float3 mergedColor = direct + indirect + specular;
        a_Output->SetPixel(mergedColor, i, 0);
    }
}

GPU_ONLY void DLSS()
{
}

GPU_ONLY void PostProcessingEffects()
{
}

CPU_GPU void WriteToOutput(int a_NumPixels, const uint2& a_Dimensions, PixelBuffer* const a_Input, uchar4* a_Output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < a_NumPixels; i += stride)
    {
        auto color = make_color(a_Input->m_Pixels[i]);
        a_Output[i] = color;
    }
}

CPU_GPU void GenerateRay(int a_NumRays, RayBatch* const a_Buffer, const float3& a_U, const float3& a_V,
                         const float3& a_W,
                         const float3& a_Eye, const int2& a_Dimensions)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < a_NumRays; i += stride)
    {
        //Convert the index into the screen dimensions.
        const int screenY = i / a_Dimensions.x;
        const int screenX = i - (screenY * a_Dimensions.x);

        float3 direction = make_float3(static_cast<float>(screenX) / a_Dimensions.x,
                                       static_cast<float>(screenY) / a_Dimensions.y, 0.f);
        float3 origin = a_Eye;

        direction.x = -(direction.x * 2.0f - 1.0f);
        direction.y = -(direction.y * 2.0f - 1.0f);
        direction = normalize(direction.x * a_U + direction.y * a_V + a_W);

        RayData ray { origin, direction, make_float3(1.f, 1.f, 1.f) };
        a_Buffer->SetRay(ray, i, 0);
    }
}