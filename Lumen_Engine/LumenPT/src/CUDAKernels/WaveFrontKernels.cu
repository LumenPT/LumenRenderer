#include "WaveFrontKernels.cuh"
#include "../Shaders/CppCommon/RenderingUtility.h"
#include "../Framework/CudaUtilities.h"

#include "../../vendor/Include/Cuda/cuda/helpers.h"
#include "../../vendor/Include/Cuda/device_launch_parameters.h"
#include "../../vendor/Include/sutil/vec_math.h"
#include "../../vendor/Include/Cuda/curand.h"
#include <cstdio>

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

    GenerateRay <<<numBlocks, blockSize>>> (numRays, a_SetupParams.m_PrimaryRays, u, v, w, eye, dimensions);

    cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;

}

CPU_ONLY void GenerateMotionVectors()
{
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

    ShadeIndirect<<<numBlocks,blockSize>>>(
        a_ShadingParams.m_ResolutionAndDepth, 
        a_ShadingParams.m_CurrentIntersections, 
        a_ShadingParams.m_CurrentRays, 
        a_ShadingParams.m_SecondaryRays);

    cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;

     //Generate shadow rays for specular highlights.
    ShadeSpecular<<<numBlocks, blockSize >>>();

    cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;

    //Generate shadow rays for direct lights.
    ShadeDirect<<<numBlocks, blockSize>>>(
        a_ShadingParams.m_ResolutionAndDepth, 
        a_ShadingParams.m_CurrentRays, 
        a_ShadingParams.m_CurrentIntersections, 
        a_ShadingParams.m_ShadowRaysBatch, 
        a_ShadingParams.m_LightBuffer, 
        a_ShadingParams.m_CDF);

    cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;

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
    Denoise<<<numBlocks, blockSize >>>();

    cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;

    MergeLightChannels <<<numBlocks, blockSize>>> (
        a_PostProcessParams.m_Resolution, 
        a_PostProcessParams.m_WavefrontOutput,
        a_PostProcessParams.m_MergedResults);

    cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;

    //TODO steal hidden Nvidia technology by breaking into their buildings
    //TODO copy merged results into image output after having run DLSS.
    DLSS<<<numBlocks, blockSize>>>();

    const int numPixelsUpscaled = a_PostProcessParams.m_UpscaledResolution.x * a_PostProcessParams.m_UpscaledResolution.y;
    const int blockSizeUpscaled = 256;
    const int numBlocksUpscaled = (numPixelsUpscaled + blockSizeUpscaled - 1) / blockSizeUpscaled;

    cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;

    //TODO
    PostProcessingEffects<<<numBlocksUpscaled, blockSizeUpscaled >>>();

    cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;

    //TODO This is temporary till the post-processing is  in place. Let the last stage copy it directly to the output buffer.
    WriteToOutput<<<numBlocksUpscaled, blockSizeUpscaled >>>(
        a_PostProcessParams.m_UpscaledResolution, 
        a_PostProcessParams.m_MergedResults, 
        a_PostProcessParams.m_ImageOutput);

    cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;

}



CPU_GPU void GenerateRay(
    int a_NumRays,
    RayBatch* const a_Buffer,
    float3 a_U,
    float3 a_V,
    float3 a_W,
    float3 a_Eye,
    int2 a_Dimensions)
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

        RayData ray{ origin, direction, make_float3(1.f, 1.f, 1.f) };
        a_Buffer->SetRay(ray, i, 0);

    }
}

CPU_GPU void ShadeDirect(
    const uint3 a_ResolutionAndDepth, 
	const RayBatch* const a_CurrentRays,
    const IntersectionBuffer* const a_CurrentIntersections, 
    ShadowRayBatch* const a_ShadowRays,
    const LightBuffer* const a_Lights,
    CDF* const a_CDF)
{

    const unsigned int numPixels = a_ResolutionAndDepth.x * a_ResolutionAndDepth.y;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    float3 potRadiance = { 0.5f,0.5f,0.5f };

    //keep CDF pointer here
	//extract light index from CDF. just random float or smth

	//PDF probability density function

	//calculate potential radiance vector from PDF and BRDF

    //Handles cases where there are less threads than there are pixels.
    //i becomes index and is to be used by in functions where you need the pixel index.
    //i will update to a new pixel index if there is less threads than there are pixels.
    for(unsigned int i = index; i < numPixels; i += stride) 
    {
        // Get intersection.
        const IntersectionData& currIntersection = a_CurrentIntersections->GetIntersection(i, 0 /*There is only one ray per pixel, only one intersection per pixel (for now)*/);
        // Get ray used to calculate intersection.   
        const RayData& currRay = a_CurrentRays->GetRay(i, 0 /*Ray index is 0 as there is only one ray per pixel*/);

        unsigned int lightIndex = 0;
        float lightPDF = 0;
        float random = RandomFloat(lightIndex);

        auto& diffColor = currIntersection.m_Primitive->m_Material->m_DiffuseColor;
    	
        potRadiance = {
            diffColor.x,
            diffColor.y,
            diffColor.z
        };

        ////pass in random number into CDF
        //get PDF from it
        //if(a_CDF != nullptr) a_CDF currently causes errors as it is not being passed in as a valid shading launch parameter.
        //{
        //    a_CDF->Get(random, lightIndex, lightPDF);
        //}

    	//World space??
        float3 sRayOrigin = currRay.m_Origin + (currRay.m_Direction * currIntersection.m_IntersectionT);

        const auto& light = a_Lights[lightIndex];

        //placeholder device primitive pointer

    	//worldspace position of emissive triangle
        const float3 lightPos = (light.m_Lights->p0 + light.m_Lights->p1 + light.m_Lights->p2) * 0.33f;
        float3 sRayDir = normalize(lightPos - sRayOrigin);

        unsigned int x, y, z;
        sRayDir = { RandomFloat(x), RandomFloat(y), RandomFloat(z) };
        sRayDir = normalize(sRayDir);
    	
        // apply epsilon... very hacky, very temporary
        sRayOrigin = sRayOrigin + (sRayDir * 0.001f);
    	
    	//Temporary forloop. Obviously not how you would wanna figure out which lights to sample
   //     for (int i = 0; i < a_Lights->m_Size; i++) // choose arbitrary light (temp) to figure out direction it.
   // 	{
   //         auto& l = a_Lights[i];
   //     	// light vertices are in world space
   //     	//????????? idk wtf????? this may not work ????? to find light position in world space???? uhhhh???? heh?
   //         float3 pos = (l.m_Lights->p0 + l.m_Lights->p1 + l.m_Lights->p2) * 0.33f;
			//sRayDir = normalize(pos - sRayOrigin);
   // 	}


        //check for fallof check what color the potential radiance is
    		//check if there is something related to contribution calculation
    	
        ShadowRayData shadowRay(
            sRayOrigin,
            sRayDir,
            1000.f,
            potRadiance,    // light contribution value at intersection point that will reflect in the direction of the incoming ray (intersection ray).
            ResultBuffer::OutputChannel::DIRECT
        );

        a_ShadowRays->SetShadowRay(shadowRay, a_ResolutionAndDepth.z, i, 0 /*Ray index is 0 as there is only one ray per pixel*/);
    }
}

CPU_GPU void ShadeSpecular()
{
}

CPU_GPU void ShadeIndirect(
    const uint3 a_ResolutionAndDepth, 
    const IntersectionBuffer* const a_Intersections, 
    const RayBatch* const a_PrimaryRays, 
    RayBatch* const a_Output)
{

    const unsigned int numPixels = a_ResolutionAndDepth.x * a_ResolutionAndDepth.y;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (int i = index; i < numPixels; i += stride)
    {
    
        //Convert the index into the screen dimensions.
        const int screenY = i / a_ResolutionAndDepth.x;
        const int screenX = i - (screenY * a_ResolutionAndDepth.x);
    
        const IntersectionData& intersection = a_Intersections->GetIntersection(i, 0);
;       const RayData& ray = a_PrimaryRays->GetRay(i, 0);
    
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
        float3 totalLightTransport = russianRouletteWeight * ray.m_Contribution * brdf;  //Total amount of light that will end up in the camera through this path.
    
        //Add to the output buffer.
        a_Output->SetRay({ pos, dir, totalLightTransport }, i, 0);
    }

}



CPU_GPU void Denoise()
{
}

CPU_GPU void MergeLightChannels(
    const uint2 a_Resolution, 
    const ResultBuffer* const a_Input, 
    PixelBuffer* const a_Output)
{

    const unsigned int numPixels = a_Resolution.x * a_Resolution.y;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (int i = index; i < numPixels; i += stride)
    {
    
        //Convert the index into the screen dimensions.
        const int screenY = i / a_Resolution.x;
        const int screenX = i - (screenY * a_Resolution.x);
    
        //Mix the results;
        float3 mergedColor = a_Input->GetPixelCombined(i);
        a_Output->SetPixel(make_float3(0.f), i, 0);
    }

}

CPU_GPU void DLSS()
{
}

CPU_GPU void PostProcessingEffects()
{
}

CPU_GPU void WriteToOutput(
    const uint2 a_Resolution, 
    const PixelBuffer* const a_Input, 
    uchar4* a_Output)
{
    const unsigned int numPixels = a_Resolution.x * a_Resolution.y;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < numPixels; i += stride)
    {
        const auto color = make_color(a_Input->GetPixel(i, 0 /*Only one channel per pixel in the merged result*/ ));
        a_Output[i] = color;
    }
}