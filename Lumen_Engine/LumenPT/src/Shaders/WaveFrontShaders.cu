#include "CppCommon/WaveFrontDataStructs.h"
#include "CppCommon/RenderingUtility.h"

#include "../../vendor/Include/Cuda/cuda/helpers.h"
#include "../../vendor/Include/sutil/vec_math.h"
#include "../../vendor/Include/Optix/optix.h"
#include "../../vendor/Include/Optix/optix_device.h"



template<typename T>
__device__ __forceinline__ static T* UnpackPointer(unsigned int a_Upper, unsigned int a_Lower)
{

    const unsigned long long ptr = static_cast<unsigned long long>(a_Upper) << 32 | a_Lower;

    return reinterpret_cast<T*>(ptr);

}

template<typename T>
__device__ __forceinline__ static void PackPointer(T const* const a_Ptr, unsigned int& a_Upper, unsigned int& a_Lower)
{

    const unsigned long long ptr = reinterpret_cast<unsigned long long>(a_Ptr);

    a_Upper = ptr >> 32;
    a_Lower = ptr & 0x00000000ffffffff;

}

extern "C"
{

    __constant__ WaveFront::OptixLaunchParameters launchParams;

}

extern "C"
__global__ void __raygen__ResolveRaysRayGen()
{

    WaveFront::OptixRayGenData* rayGenData = reinterpret_cast<WaveFront::OptixRayGenData*>(optixGetSbtDataPointer());
    const float minDistance = rayGenData->m_MinDistance;
    const float maxDistance = rayGenData->m_MaxDistance;

    //For each pixel:
    //1. Get ray definition from buffer
    const uint3 launchIndex = optixGetLaunchIndex();
    const uint3 launchDim = optixGetLaunchDimensions();

    const unsigned numPixels = launchDim.x * launchDim.y;
    const unsigned numPixelsBuffer = launchParams.m_IntersectionRayBatch->m_NumPixels;
    const unsigned maxPixelBufferSize = launchParams.m_IntersectionRayBatch->GetSize();

    const unsigned pixelIndex = launchIndex.y * launchDim.x + launchIndex.x;
    const unsigned sampleIndex = launchIndex.z;

    const unsigned int rayArrayIndex = launchParams.m_IntersectionRayBatch->GetRayArrayIndex(pixelIndex, sampleIndex);
    const WaveFront::IntersectionRayData& rayData = launchParams.m_IntersectionRayBatch->GetRay(rayArrayIndex);

    //2. Trace ray: optixTrace()

    WaveFront::IntersectionData intersection{};
    intersection.m_RayArrayIndex = rayArrayIndex;

    unsigned int intersectionPtr_Up = 0;
    unsigned int intersectionPtr_Low = 0;

    PackPointer(&intersection, intersectionPtr_Up, intersectionPtr_Low);

    const OptixTraversableHandle scene = launchParams.m_TraversableHandle;

    if(rayData.IsValidRay())
    {

        optixTrace(
            scene,
            rayData.m_Origin,
            rayData.m_Direction,
            minDistance,
            maxDistance,
            0.f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0, //SBT offset for selecting the SBT records to use
            1, //SBT stride for selecting the SBT records to use, multiplied with SBT-GAS index
            0,
            intersectionPtr_Up,
            intersectionPtr_Low);

    }

    //3. Store IntersectionData in buffer

    launchParams.m_IntersectionBuffer->SetIntersection(intersection, pixelIndex, launchIndex.z);

    return;

}

extern"C"
__global__ void __closesthit__ResolveRaysClosestHit()
{

    DevicePrimitive* hitPrimitive = reinterpret_cast<DevicePrimitive*>(optixGetSbtDataPointer());
    
    //If closest hit found, return IntersectionData.
    
    const unsigned int intersectionPtr_Up = optixGetPayload_0();
    const unsigned int intersectionPtr_Low = optixGetPayload_1();

    
    WaveFront::IntersectionData* intersection = UnpackPointer<WaveFront::IntersectionData>(intersectionPtr_Up, intersectionPtr_Low);
    
    intersection->m_IntersectionT = optixGetRayTmax();
    intersection->m_UVs = optixGetTriangleBarycentrics(); //Unnecessary, can be get from vertex data inside the  m_Primitive.
    intersection->m_PrimitiveIndex = optixGetPrimitiveIndex();
    intersection->m_Primitive = hitPrimitive;

    /*printf("Shader - Primitive: %p, m_IndexBuffer: %p, m_VertexBuffer: %p \n",
        hitPrimitive,
        hitPrimitive->m_IndexBuffer,
        hitPrimitive->m_VertexBuffer);*/

    return;

}

extern"C"
__global__ void __miss__ResolveRaysMiss()
{

    return;

}

extern "C"
__global__ void __raygen__ResolveShadowRaysRayGen()
{

    WaveFront::OptixRayGenData* rayGenData = reinterpret_cast<WaveFront::OptixRayGenData*>(optixGetSbtDataPointer());
    const float minDistance = rayGenData->m_MinDistance;
    const float maxDistance = rayGenData->m_MaxDistance;

    //For each pixel:
    //For depth (num rays per pixel)
    //1. Get shadow ray definition from buffer

    const unsigned int numRaysPerPixel = launchParams.m_ShadowRayBatch->m_RaysPerPixel;
    const unsigned int maxDepth = launchParams.m_ShadowRayBatch->m_MaxDepth;

    const uint3 launchIndex = optixGetLaunchIndex();
    const uint3 launchDim = optixGetLaunchDimensions();

    const unsigned int pixelIndex = launchDim.x * launchIndex.y + launchIndex.x;
    const unsigned int depthIndex = launchIndex.z;
    const unsigned int rayIndex = 0;

    /*printf("LaunchIndex: (%i, %i, %i) LaunchDim: (%i, %i, %i) PixelIndex: %i, DepthIndex: %i \n",
        launchIndex.x, launchIndex.y, launchIndex.z,
        launchDim.x, launchDim.y, launchDim.z,
        pixelIndex,
        depthIndex);*/

    const unsigned int rayArrayIndex = launchParams.m_ShadowRayBatch->GetShadowRayArrayIndex(depthIndex, pixelIndex, rayIndex);
    const WaveFront::ShadowRayData& rayData = launchParams.m_ShadowRayBatch->GetShadowRayData(rayArrayIndex);

    //2. Trace ray: optixTrace()

    const OptixTraversableHandle scene = launchParams.m_TraversableHandle;

    WaveFront::OptixOcclusion occlusion(rayData.m_MaxDistance);

    unsigned int occlusionPtr_Up = 0;
    unsigned int occlusionPtr_Low = 0;

    PackPointer(&occlusion, occlusionPtr_Up, occlusionPtr_Low);

    if(rayData.IsValidRay())
    {

        optixTrace(
            scene,
            rayData.m_Origin,
            rayData.m_Direction,
            minDistance,
            maxDistance,
            0.f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0,
            1,
            0,
            occlusionPtr_Up,
            occlusionPtr_Low);

    }

    ////3. If no hit, accumulate result in buffer;

    float3 red = make_float3(1.f, 0.f, 1.f);
    float3 blue = make_float3(0.f, 0.f, 1.f);

    if(!occlusion.m_Occluded)
    {
        /*resolveShadowRaysParams.m_Results->SetPixel(rayData.m_PotentialRadiance, pixelIndex, rayData.m_OutputChannelIndex);

        float3 setColor = resolveShadowRaysParams.m_Results->m_PixelBuffer->GetPixel(pixelIndex, static_cast<unsigned>(rayData.m_OutputChannelIndex));

        const unsigned pixelArrIndex = resolveShadowRaysParams.m_Results->m_PixelBuffer->GetPixelArrayIndex(pixelIndex, static_cast<unsigned>(rayData.m_OutputChannelIndex));

        const float3* pixelPtr = &resolveShadowRaysParams.m_Results->m_PixelBuffer->m_Pixels[pixelArrIndex];

        if(setColor.x != 0.f || setColor.y != 0.f || setColor.z != 0.f)
        {
            printf("PixelIndex:%i PixelPtr: %p ColorSet: %f, %f, %f \n", pixelIndex, pixelPtr, setColor.x, setColor.y, setColor.z);
        }*/

        launchParams.m_ResultBuffer->SetPixel(red, pixelIndex, WaveFront::ResultBuffer::OutputChannel::DIRECT);

    }
    else
    {
        launchParams.m_ResultBuffer->SetPixel(blue, pixelIndex, WaveFront::ResultBuffer::OutputChannel::DIRECT);
    }

    return;

}

extern "C"
__global__ void __anyhit__ResolveShadowRaysAnyHit()
{

    //If any hit found before max distance, return true;

    const unsigned int intersectionPtr_Up = optixGetPayload_0();
    const unsigned int intersectionPtr_Low = optixGetPayload_1();

    WaveFront::OptixOcclusion* occlusion = UnpackPointer<WaveFront::OptixOcclusion>(intersectionPtr_Up, intersectionPtr_Low);

    const float intersectionT = optixGetRayTmax();

    if (intersectionT < occlusion->m_MaxDistance)
    {
        occlusion->m_Occluded = true;
        optixTerminateRay();
        return;
    }

    return;

}

extern "C"
__global__ void __miss__ResolveShadowRaysMiss()
{

    const unsigned int intersectionPtr_Up = optixGetPayload_0();
    const unsigned int intersectionPtr_Low = optixGetPayload_1();

    WaveFront::OptixOcclusion* occlusion = UnpackPointer<WaveFront::OptixOcclusion>(intersectionPtr_Up, intersectionPtr_Low);

    occlusion->m_Occluded = false;

    return;

}