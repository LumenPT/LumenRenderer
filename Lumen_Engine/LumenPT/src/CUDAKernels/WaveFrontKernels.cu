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
    /*ShadeDirect<<<numBlocks, blockSize>>>(
        a_ShadingParams.m_ResolutionAndDepth, 
        a_ShadingParams.m_CurrentRays, 
        a_ShadingParams.m_CurrentIntersections, 
        a_ShadingParams.m_ShadowRaysBatch, 
        a_ShadingParams.m_LightBuffer, 
        a_ShadingParams.m_CDF);*/

    DEBUGShadePrimIntersections <<<numBlocks, blockSize >> > (
        a_ShadingParams.m_ResolutionAndDepth,
        a_ShadingParams.m_CurrentRays,
        a_ShadingParams.m_CurrentIntersections,
        a_ShadingParams.m_DEBUGResultBuffer);

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
    //Denoise<<<numBlocks, blockSize >>>();

    /*cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;*/

    MergeLightChannels <<<numBlocks, blockSize>>> (
        a_PostProcessParams.m_Resolution, 
        a_PostProcessParams.m_WavefrontOutput,
        a_PostProcessParams.m_MergedResults);

    /*cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;*/

    //TODO steal hidden Nvidia technology by breaking into their buildings
    //TODO copy merged results into image output after having run DLSS.
    DLSS<<<numBlocks, blockSize>>>();

    const int numPixelsUpscaled = a_PostProcessParams.m_UpscaledResolution.x * a_PostProcessParams.m_UpscaledResolution.y;
    const int blockSizeUpscaled = 256;
    const int numBlocksUpscaled = (numPixelsUpscaled + blockSizeUpscaled - 1) / blockSizeUpscaled;

    /*cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;*/

    //TODO
    PostProcessingEffects<<<numBlocksUpscaled, blockSizeUpscaled >>>();

    /*cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;*/

    //TODO This is temporary till the post-processing is  in place. Let the last stage copy it directly to the output buffer.
    WriteToOutput<<<numBlocksUpscaled, blockSizeUpscaled >>>(
        a_PostProcessParams.m_UpscaledResolution, 
        a_PostProcessParams.m_MergedResults, 
        a_PostProcessParams.m_ImageOutput);

    cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;

}



CPU_ON_GPU void GenerateRay(
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

CPU_ON_GPU void ShadeDirect(
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

        if(currIntersection.IsIntersection())
        {

            // Get ray used to calculate intersection.
            const unsigned int rayArrayIndex = currIntersection.m_RayArrayIndex;

            const RayData& currRay = a_CurrentRays->GetRay(rayArrayIndex);
            
            //unsigned int lightIndex = 0;
            //float lightPDF = 0;
            //float random = RandomFloat(lightIndex);

            //auto& diffColor = currIntersection.m_Primitive->m_Material->m_DiffuseColor;
            
            //potRadiance = make_float3(diffColor);

            potRadiance = make_float3(1.f);

            float3 sRayOrigin = currRay.m_Origin + (currRay.m_Direction * currIntersection.m_IntersectionT);
            
            //const auto& light = a_Lights[lightIndex];
            //
            ////worldspace position of emissive triangle
            //const float3 lightPos = (light.m_Lights->p0 + light.m_Lights->p1 + light.m_Lights->p2) * 0.33f;
            //float3 sRayDir = normalize(lightPos - sRayOrigin);


            unsigned int x = threadIdx.x , y = threadIdx.y, z = threadIdx.z;
            float3 sRayDir = { RandomFloat(x), RandomFloat(y), RandomFloat(z) };
            sRayDir = normalize(sRayDir);
            
            // apply epsilon... very hacky, very temporary
            sRayOrigin = sRayOrigin + (sRayDir * 0.001f);

            ShadowRayData shadowRay(
                sRayOrigin,
                sRayDir,
                0.11f,
                potRadiance,    // light contribution value at intersection point that will reflect in the direction of the incoming ray (intersection ray).
                ResultBuffer::OutputChannel::DIRECT
            );
            
            a_ShadowRays->SetShadowRay(shadowRay, a_ResolutionAndDepth.z, i, 0 /*Ray index is 0 as there is only one ray per pixel*/);

        }
        
    }
}

CPU_ON_GPU void ShadeSpecular()
{
}

CPU_ON_GPU void ShadeIndirect(
    const uint3 a_ResolutionAndDepth, 
    const RayBatch* const a_PrimaryRays,  //TODO: These need to be the rays of that the current wave intersected before calling this wave.
    const IntersectionBuffer* const a_Intersections, 
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

CPU_ON_GPU void DEBUGShadePrimIntersections(
    const uint3 a_ResolutionAndDepth,
    const RayBatch* const a_PrimaryRays,
    const IntersectionBuffer* const a_PrimaryIntersections,
    ResultBuffer* const a_Output)
{

    const unsigned int numPixels = a_ResolutionAndDepth.x * a_ResolutionAndDepth.y;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < numPixels; i += stride)
    {

        // Get intersection.
        const IntersectionData& currIntersection = a_PrimaryIntersections->GetIntersection(i, 0 /*There is only one ray per pixel, only one intersection per pixel (for now)*/);

        if (currIntersection.IsIntersection())
        {

            // Get ray used to calculate intersection.
            const unsigned int rayArrayIndex = currIntersection.m_RayArrayIndex;

            const unsigned int vertexIndex = 3 * currIntersection.m_PrimitiveIndex;
            const DevicePrimitive* primitive = currIntersection.m_Primitive;

            if( primitive == nullptr || 
                primitive->m_IndexBuffer == nullptr || 
                primitive->m_VertexBuffer == nullptr ||
                primitive->m_Material == nullptr)
            {

                if(!primitive)
                {
                    printf("Error, primitive: %p \n", primitive);
                }
                else
                {
                    printf("Error, found nullptr in primitive variables: \n\tm_IndexBuffer: %p \n\tm_VertexBuffer: %p \n\tm_Material: %p\n",
                        primitive->m_IndexBuffer,
                        primitive->m_VertexBuffer,
                        primitive->m_Material);
                }
                
                a_Output->SetPixel(make_float3(1.f, 0.f, 1.f), rayArrayIndex, ResultBuffer::OutputChannel::DIRECT);
                return;
            }

            /*printf("VertexIndex: %i, Primitive: %p, m_IndexBuffer: %p, m_VertexBuffer: %p \n", 
                vertexIndex, 
                primitive, 
                primitive->m_IndexBuffer, 
                primitive->m_VertexBuffer);*/

            const unsigned int vertexIndexA = primitive->m_IndexBuffer[vertexIndex + 0];
            const unsigned int vertexIndexB = primitive->m_IndexBuffer[vertexIndex + 1];
            const unsigned int vertexIndexC = primitive->m_IndexBuffer[vertexIndex + 2];

            const Vertex* A = &primitive->m_VertexBuffer[vertexIndexA];
            const Vertex* B = &primitive->m_VertexBuffer[vertexIndexB];
            const Vertex* C = &primitive->m_VertexBuffer[vertexIndexC];

            const float U = currIntersection.m_UVs.x;
            const float V = currIntersection.m_UVs.y;
            const float W = 1.f - (U + V);

            const float2 texCoords = A->m_UVCoord * W + B->m_UVCoord * U + C->m_UVCoord * V;

            if(U + V + W != 1.f || texCoords.x > 1.f || texCoords.y > 1.f)
            {

                if(U + V + W != 1.f)
                {
                    printf("U: %f, V: %f, W: %f \n", U, V, W);
                    a_Output->SetPixel(make_float3(1.f, 1.f, 0.f), rayArrayIndex, ResultBuffer::OutputChannel::DIRECT);
                }
                else
                {
                    //printf("X: %f, Y: %f \n", texCoords.x, texCoords.y);
                    a_Output->SetPixel(make_float3(0.f, 1.f, 1.f), rayArrayIndex, ResultBuffer::OutputChannel::DIRECT);
                }
                return;
                
            }

            const DeviceMaterial* material = primitive->m_Material;

            const float4 textureColor = tex2D<float4>(material->m_DiffuseTexture, texCoords.x, texCoords.y);
            const float3 finalColor = make_float3(textureColor * material->m_DiffuseColor);

            a_Output->SetPixel(finalColor, rayArrayIndex, ResultBuffer::OutputChannel::DIRECT);

        }

    }

}



CPU_ON_GPU void Denoise()
{
}

CPU_ON_GPU void MergeLightChannels(
    const uint2 a_Resolution, 
    const ResultBuffer* const a_Input, 
    PixelBuffer* const a_Output)
{

    const unsigned int numPixels = a_Resolution.x * a_Resolution.y;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    const float3* firstPixel = a_Input->m_PixelBuffer->m_Pixels;

    const unsigned int numPixelsBuffer = a_Input->m_PixelBuffer->m_NumPixels;
    const unsigned int numChannelsPixelsBuffer = a_Input->m_PixelBuffer->m_ChannelsPerPixel;

    /*printf("NumPixelsBuffer: %i NumChannelsPixelsBuffer: %i FirstPixelPtr: %p PixelBufferPtr: %p \n", 
        numPixelsBuffer, numChannelsPixelsBuffer, firstPixel, a_Input->m_PixelBuffer);*/

    for (int i = index; i < numPixels; i += stride)
    {
    
        //Convert the index into the screen dimensions.
        const int screenY = i / a_Resolution.x;
        const int screenX = i - (screenY * a_Resolution.x);

        //Mix the results;
        float3 mergedColor = a_Input->GetPixelCombined(i);
        a_Output->SetPixel(mergedColor, i, 0);

        //printf("MergedColor: %f, %f, %f \n", mergedColor.x, mergedColor.y, mergedColor.z);

    }

}

CPU_ON_GPU void DLSS()
{
}

CPU_ON_GPU void PostProcessingEffects()
{
}

CPU_ON_GPU void WriteToOutput(
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

//Data buffer helper functions

//Reset ray batch

CPU_ONLY void ResetRayBatch(
    RayBatch* const a_RayBatchDevPtr, 
    unsigned int a_NumPixels, 
    unsigned int a_RaysPerPixel)
{

    ResetRayBatchMembers <<<1,1>>> (a_RayBatchDevPtr, a_NumPixels, a_RaysPerPixel);

    const int numRays = a_NumPixels * a_RaysPerPixel;
    const int blockSize = 256;
    const int numBlocks = (numRays + blockSize - 1) / blockSize;

    ResetRayBatchData<<<numBlocks,blockSize>>>(a_RayBatchDevPtr);

}

CPU_ON_GPU void ResetRayBatchMembers(
    RayBatch* const a_RayBatch, 
    unsigned int a_NumPixels, 
    unsigned int a_RaysPerPixel)
{
    *const_cast<unsigned*>(&a_RayBatch->m_NumPixels) = a_NumPixels;
    *const_cast<unsigned*>(&a_RayBatch->m_RaysPerPixel) = a_RaysPerPixel;

}

CPU_ON_GPU void ResetRayBatchData(RayBatch* const a_RayBatch)
{

    const unsigned int bufferSize = a_RayBatch->GetSize();
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < bufferSize; i+=stride)
    {
        a_RayBatch->m_Rays[i] = RayData{};
    }

}

//Reset shadow ray batch

CPU_ONLY void ResetShadowRayBatch(
    ShadowRayBatch* a_ShadowRayBatchDevPtr,
    unsigned int a_MaxDepth,
    unsigned int a_NumPixels,
    unsigned int a_RaysPerPixel)
{

    ResetShadowRayBatchMembers<<<1,1>>>(a_ShadowRayBatchDevPtr, a_MaxDepth, a_NumPixels, a_RaysPerPixel);

    const int numRays = a_MaxDepth * a_NumPixels * a_RaysPerPixel;
    const int blockSize = 256;
    const int numBlocks = (numRays + blockSize - 1) / blockSize;

    ResetShadowRayBatchData<<<numBlocks, blockSize>>>(a_ShadowRayBatchDevPtr);

}

CPU_ON_GPU void ResetShadowRayBatchMembers(
    ShadowRayBatch* const a_ShadowRayBatch,
    unsigned int a_MaxDepth,
    unsigned int a_NumPixels,
    unsigned int a_RaysPerPixel)
{

    *const_cast<unsigned*>(&a_ShadowRayBatch->m_MaxDepth) = a_MaxDepth;
    *const_cast<unsigned*>(&a_ShadowRayBatch->m_NumPixels) = a_NumPixels;
    *const_cast<unsigned*>(&a_ShadowRayBatch->m_RaysPerPixel) = a_RaysPerPixel;

}

CPU_ON_GPU void ResetShadowRayBatchData(ShadowRayBatch* const a_ShadowRayBatch)
{

    const unsigned int bufferSize = a_ShadowRayBatch->GetSize();
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for(unsigned int i = index; i < bufferSize; i += stride)
    {

        a_ShadowRayBatch->m_ShadowRays[i] = ShadowRayData{};
    }

}

//Reset pixel buffer

CPU_ONLY void ResetPixelBuffer(
    PixelBuffer* a_PixelBufferDevPtr, 
    unsigned a_NumPixels, 
    unsigned a_ChannelsPerPixel)
{

    ResetPixelBufferMembers<<<1, 1>>>(a_PixelBufferDevPtr, a_NumPixels, a_ChannelsPerPixel);

    const int totalPixels = a_NumPixels * a_ChannelsPerPixel;
    const int blockSize = 256;
    const int numBlocks = (totalPixels + blockSize - 1) / blockSize;

    ResetPixelBufferData <<<numBlocks, blockSize>>>(a_PixelBufferDevPtr);

}

CPU_ON_GPU void ResetPixelBufferMembers(
    PixelBuffer* const a_PixelBuffer, 
    unsigned a_NumPixels, 
    unsigned a_ChannelsPerPixel)
{

    *const_cast<unsigned*>(&a_PixelBuffer->m_NumPixels) = a_NumPixels;
    *const_cast<unsigned*>(&a_PixelBuffer->m_ChannelsPerPixel) = a_ChannelsPerPixel;

}

CPU_ON_GPU void ResetPixelBufferData(PixelBuffer* const a_PixelBuffer)
{

    const unsigned int bufferSize = a_PixelBuffer->GetSize();
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < bufferSize; i += stride)
    {

        a_PixelBuffer->m_Pixels[i] = { 0.f, 0.f, 0.f };
    }

}
