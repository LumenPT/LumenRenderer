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
__device__  void ResolveReSTIRRayGen(const uint3 a_LaunchIndex)
{
    //Launch as a 1D Array so that idx.x corresponds to the literal ray index.
    const uint3 idx = optixGetLaunchIndex();

    //Retrieve the data.
    //const RestirShadowRay& rayData = reSTIRParams.shadowRays[idx.x];
    //const OptixTraversableHandle scene = reSTIRParams.optixSceneHandle;
    //auto reservoirIndex = rayData.index;

    //optixTrace(
    //    scene,
    //    rayData.origin,
    //    rayData.direction,
    //    0.005f, //Prevent self shadowing so offset a little bit.
    //    rayData.distance,   //Max distance already has a small offset to prevent self-shadowing.
    //    0.f,
    //    OptixVisibilityMask(255),
    //    OPTIX_RAY_FLAG_NONE,
    //    0,  //TODO
    //    1,  //TODO
    //    0,  //TODO
    //    reservoirIndex  //Pass the reservoir index so that it can be set to 0 when a hit is found.
    //);

    return;
}

extern "C"
__device__ void ResolveRaysRayGen(const uint3 a_LaunchIndex)
{
    WaveFront::OptixRayGenData* rayGenData = reinterpret_cast<WaveFront::OptixRayGenData*>(optixGetSbtDataPointer());
    const float minDistance = rayGenData->m_MinDistance;
    const float maxDistance = rayGenData->m_MaxDistance;

    //For each pixel:
    //1. Get ray definition from buffer
    const unsigned sampleIndex = 0;
    const unsigned int rayArrayIndex = launchParams.m_IntersectionRayBatch->GetRayArrayIndex(a_LaunchIndex.x, sampleIndex);
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
    launchParams.m_IntersectionBuffer->SetIntersection(intersection, a_LaunchIndex.x, a_LaunchIndex.z);

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

extern "C"
__device__ void ResolveShadowRaysRayGen(const uint3 a_LaunchIndex)
{
    WaveFront::OptixRayGenData* rayGenData = reinterpret_cast<WaveFront::OptixRayGenData*>(optixGetSbtDataPointer());
    const float minDistance = rayGenData->m_MinDistance;
    const float maxDistance = rayGenData->m_MaxDistance;

    //For each pixel:
    //For depth (num rays per pixel)
    //1. Get shadow ray definition from buffer
    const unsigned int depthIndex = a_LaunchIndex.z;
    const unsigned int rayIndex = 0;

    /*printf("LaunchIndex: (%i, %i, %i) LaunchDim: (%i, %i, %i) PixelIndex: %i, DepthIndex: %i \n",
        launchIndex.x, launchIndex.y, launchIndex.z,
        launchDim.x, launchDim.y, launchDim.z,
        pixelIndex,
        depthIndex);*/

    const unsigned int rayArrayIndex = launchParams.m_ShadowRayBatch->GetShadowRayArrayIndex(depthIndex, a_LaunchIndex.x, rayIndex);
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

        launchParams.m_ResultBuffer->SetPixel(red, a_LaunchIndex.x, WaveFront::ResultBuffer::OutputChannel::DIRECT);

    }
    else
    {
        launchParams.m_ResultBuffer->SetPixel(blue, a_LaunchIndex.x, WaveFront::ResultBuffer::OutputChannel::DIRECT);
    }

    return;

}

__global__ void __anyhit__UberAnyHit()
{
    switch (launchParams.m_TraceType)
    {
    case WaveFront::RayType::INTERSECTION_RAY:
        break;
    case WaveFront::RayType::SHADOW_RAY:
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
        break;
    case WaveFront::RayType::RESTIR_RAY:
        break;
    }
}

__global__ void __miss__UberMiss()
{
    switch (launchParams.m_TraceType)
    {
    case WaveFront::RayType::INTERSECTION_RAY:
        return;
        break;
    case WaveFront::RayType::SHADOW_RAY:
        const unsigned int intersectionPtr_Up = optixGetPayload_0();
        const unsigned int intersectionPtr_Low = optixGetPayload_1();
        WaveFront::OptixOcclusion* occlusion = UnpackPointer<WaveFront::OptixOcclusion>(intersectionPtr_Up, intersectionPtr_Low);
        occlusion->m_Occluded = false;
        break;
    case WaveFront::RayType::RESTIR_RAY:
        return;
        break;
    }
}

__global__ void __closesthit__UberClosestHit()
{
    switch (launchParams.m_TraceType)
    {
    case WaveFront::RayType::INTERSECTION_RAY:
        DevicePrimitive* hitPrimitive = reinterpret_cast<DevicePrimitive*>(optixGetSbtDataPointer());
        //If closest hit found, return IntersectionData.
        const unsigned int intersectionPtr_Up = optixGetPayload_0();
        const unsigned int intersectionPtr_Low = optixGetPayload_1();
        WaveFront::IntersectionData* intersection = UnpackPointer<WaveFront::IntersectionData>(intersectionPtr_Up, intersectionPtr_Low);
        intersection->m_IntersectionT = optixGetRayTmax();
        intersection->m_UVs = optixGetTriangleBarycentrics(); //Unnecessary, can be get from vertex data inside the  m_Primitive.
        intersection->m_PrimitiveIndex = optixGetPrimitiveIndex();
        intersection->m_Primitive = hitPrimitive;
        break;
    case WaveFront::RayType::SHADOW_RAY:
        return;
        break;
    case WaveFront::RayType::RESTIR_RAY:
      //Get the reservoir and set its weight to 0 so that it is no longer considered a valid candidate.
      //reSTIRParams.reservoirs[optixGetAttribute_0()].weight = 0.f;
      //optixTerminateRay();
        return;
        break;
    }
}

extern "C"
__global__ void __raygen__UberGenShader()
{
    const uint3 launchIndex = optixGetLaunchIndex();

	switch (launchParams.m_TraceType)
	{
    case WaveFront::RayType::INTERSECTION_RAY:
        //Primary rays
        ResolveRaysRayGen(launchIndex);
        break;
    case WaveFront::RayType::SHADOW_RAY:
        //Shadow rays
        ResolveShadowRaysRayGen(launchIndex);
        break;
    case WaveFront::RayType::RESTIR_RAY:
        //Actually Primary rays
        ResolveReSTIRRayGen(launchIndex);
        break;
	}
}