#include "WaveFrontKernels.cuh"
#include "../Shaders/CppCommon/RenderingUtility.h"

#include "../../vendor/Include/Cuda/cuda/helpers.h"
#include "../../vendor/Include/Cuda/device_launch_parameters.h"
#include "../../vendor/Include/sutil/vec_math.h"
#include <cstdio>

using namespace WaveFront;

CPU_ONLY void CheckLastCudaError()
{

    cudaError err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        printf("CUDA error occured: %s : %s", cudaGetErrorName(err), cudaGetErrorString(err));
        abort();
    }

}

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

    cudaDeviceSynchronize();
    CheckLastCudaError();

    GenerateRay <<<numBlocks, blockSize>>> (numRays, a_SetupParams.m_PrimaryRays, u, v, w, eye, dimensions);

    cudaDeviceSynchronize();
    CheckLastCudaError();

}



CPU_ONLY void Shade(const ShadingLaunchParameters& a_ShadingParams)
{
    //TODO
    /*
     * - Implement the below functions.
     * - Access to intersection data, as well as the ray resulting in this shading for chained BRDF scaling.
     */

     //Generate secondary rays.
    ShadeIndirect<<<1,1>>>(a_ShadingParams.m_ResolutionAndDepth, a_ShadingParams.m_CurrentIntersections, a_ShadingParams.m_CurrentRays, a_ShadingParams.m_SecondaryRays); 
     //Generate shadow rays for specular highlights.
    ShadeSpecular<<<1,1>>>();
    //Generate shadow rays for direct lights.
    ShadeDirect<<<1,1>>>(a_ShadingParams.m_ResolutionAndDepth, a_ShadingParams.m_CurrentRays, a_ShadingParams.m_CurrentIntersections, a_ShadingParams.m_ShadowRaysBatch, a_ShadingParams.m_LightBuffer);   
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
    Denoise<<<1,1>>>();

    MergeLightChannels <<<numBlocks, blockSize >>> (
        numPixels, 
        a_PostProcessParams.m_Resolution, 
        a_PostProcessParams.m_WavefrontOutput->m_PixelOutput,
        a_PostProcessParams.m_MergedResults);

    //TODO steal hidden Nvidia technology by breaking into their buildings
    //TODO copy merged results into image output after having run DLSS.
    DLSS<<<1,1>>>();

    //TODO
    PostProcessingEffects<<<1,1>>>();



    //TODO This is temporary till the post-processing is  in place. Let the last stage copy it directly to the output buffer.
    WriteToOutput<<<numBlocks, blockSize>>>(numPixels, a_PostProcessParams.m_Resolution, a_PostProcessParams.m_MergedResults, a_PostProcessParams.m_ImageOutput);
}



CPU_ONLY void GenerateMotionVectors()
{
}

CPU_GPU void ShadeDirect(
    const uint3& a_ResolutionAndDepth, 
	const RayBatch* const a_CurrentRays,
    const IntersectionBuffer* const a_Intersections, 
    ShadowRayBatch* const a_ShadowRays,
    const LightBuffer* const a_Lights)
{
    // need access to light & mesh
        // I get triangleID and meshID, so need to use that to look up actual data based on those IDs
        // 

    int pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;  //use in intersection and shadowray batches
    int stride = blockDim.x * gridDim.x;

	//max number of pixels
    const int numPixels = a_ResolutionAndDepth.x * a_ResolutionAndDepth.y;

	//outgoing shadow ray origin

	//into intersection buffer
	

    /*
     * a_ShadingParams->m_Intersections.m_Intersections[i]  //      intersection data
     *              find intersecting triangle properties through MeshID & TriangleID
     */

    float3 potRadiance = { 0.5f,0.5f,0.5f };


	const auto& currIntersection =  a_Intersections->GetIntersection(pixelIndex);
	
    if(pixelIndex < numPixels && currIntersection.IsIntersection())
    {
	    // get position of intersection
        unsigned int currRayIndex = a_CurrentRays->GetRayIndex(pixelIndex, /*Temp*/ a_CurrentRays->GetSize());   //figure out ray index
        const auto& currRay = a_CurrentRays->GetRay(pixelIndex, currRayIndex);

    	//World space??
        float3 sRayOrigin = currRay.m_Origin + (currRay.m_Direction * currIntersection.m_IntersectionT);
        float3 sRayDir {0.0f, 1.0f, 0.0f};
    	
    	//Temporary forloop. Obviously not how you would wanna figure out which lights to sample
        for (int i = 0; i < a_Lights->m_Size; i++) // choose arbitrary light (temp) to figure out direction it.
    	{
            auto& l = a_Lights[i];
        	// light vertices are in world space
			sRayDir = normalize(float3(l.) - sRayOrigin);
    	}

        // hacky epsilon value... very temporary
        sRayOrigin = sRayOrigin + (sRayDir * 0.001f);

        //check for fallof check what color the potential radiance is
    		//check if there is something related to contribution calculation
    	
        ShadowRayData shadowRay(
            sRayOrigin,
            sRayDir,
            1000.f,
            potRadiance,    // contribution/importance(?) value
            ResultBuffer::OutputChannel::DIRECT
        );

        a_ShadowRays->SetShadowRay(shadowRay, a_ResolutionAndDepth.z, pixelIndex, currRayIndex);
    	
        // using intersection buffer & current ray batch, with helper functions, which take pixel index etc.
    }
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

CPU_GPU void Denoise()
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

CPU_GPU void DLSS()
{
}

CPU_GPU void PostProcessingEffects()
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

    

    //printf("NumPixel RayBatch: %i \n", a_Buffer->GetSize());
    //printf("NumRays RayBatch: %i \n", a_Buffer->m_RaysPerPixel);

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

        //printf("Stride: %i, NumRays: %i, Pixel index: %i \n", stride, a_NumRays, i);

    }
}