#include "./CppCommon/WaveFrontKernels.cuh"

#include "CppCommon/RenderingUtility.h"

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

    //TODO pass buffer
    GenerateRay <<<numBlocks, blockSize >>> (numRays, nullptr, u, v, w, eye, dimensions);
}



CPU_ONLY void Shade(const ShadingLaunchParameters& a_ShadingParams)
{
    //TODO
    /*
     * - Implement the below functions.
     * - Access to intersection data, as well as the ray resulting in this shading for chained BRDF scaling.
     */

    ShadeIndirect<<<1,1>>>(); //Generate secondary rays.
    ShadeSpecular<<<1,1>>>(); //Generate shadow rays for specular highlights.
    ShadeDirect<<<1,1>>>();   //Generate shadow rays for direct lights.
}


CPU_ONLY void PostProcess(const PostProcessLaunchParameters& a_PostProcessParams)
{
    //TODO
    /*
     * Not needed now. Can be implemented later.
     * For now just merge the final light contributions to get the final pixel color.
     */
    Denoise();
    MergeLightChannels<<<1,1>>>();
    DLSS();
    PostProcessingEffects();
}



CPU_ONLY void GenerateMotionVectors()
{
}

CPU void ShadeDirect()
{
}

CPU void ShadeSpecular()
{
}

CPU void ShadeIndirect()
{
}

GPU void Denoise()
{
}

CPU void MergeLightChannels()
{
}

GPU void DLSS()
{
}

GPU void PostProcessingEffects()
{
}

CPU void GenerateRay(int a_NumRays, RayData* a_Buffer, const float3& a_U, const float3& a_V, const float3& a_W,
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
        a_Buffer[i] = ray;
    }
}