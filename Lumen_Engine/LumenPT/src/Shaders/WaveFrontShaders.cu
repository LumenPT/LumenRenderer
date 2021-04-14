#include "CppCommon/WaveFrontDataStructs.h"
#include "CppCommon/RenderingUtility.h"

#include "../../vendor/Include/Cuda/cuda/helpers.h"
#include "../../vendor/Include/sutil/vec_math.h"
#include "../../vendor/Include/Optix/optix.h"
#include "../../vendor/Include/Optix/optix_device.h"

extern "C"
{

    __constant__ WaveFront::OptixLaunchParameters launchParams;

}



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
__device__ __forceinline__ void IntersectionRaysRayGen()
{


 
    //1. Get ray definition from buffer
    const unsigned idx = optixGetLaunchIndex().x;
    const WaveFront::IntersectionRayData& rayData = *launchParams.m_IntersectionRayBatch->GetData(idx);

    //2. Trace ray: optixTrace()
    WaveFront::IntersectionData intersection{};
    intersection.m_RayArrayIndex = idx;
    intersection.m_PixelIndex = rayData.m_PixelIndex;

    unsigned int intersectionPtr_Up = 0;
    unsigned int intersectionPtr_Low = 0;

    PackPointer(&intersection, intersectionPtr_Up, intersectionPtr_Low);

    const OptixTraversableHandle scene = launchParams.m_TraversableHandle;

    optixTrace(
        scene,
        rayData.m_Origin,
        rayData.m_Direction,
        launchParams.m_MinMaxDistance.x,
        launchParams.m_MinMaxDistance.y,
        0.f, //Ray Time, can be 0 in our case.
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0, //SBT offset for selecting the SBT records to use
        0, //SBT stride for selecting the SBT records to use, multiplied with SBT-GAS index
        0, //Miss SBT index, always use first miss shader.
        intersectionPtr_Up,
        intersectionPtr_Low);

    //3. Store IntersectionData in buffer
    launchParams.m_IntersectionBuffer->Add(&intersection);

    return;

}

extern "C"
__device__ __forceinline__ void ShadowRaysRayGen()
{

    unsigned int idx = optixGetLaunchIndex().x;
    const WaveFront::ShadowRayData& rayData = *launchParams.m_ShadowRayBatch->GetData(idx);

    //2. Trace ray: optixTrace()
    const OptixTraversableHandle scene = launchParams.m_TraversableHandle;

    unsigned int isIntersection = 0;

    optixTrace(
        scene,
        rayData.m_Origin,
        rayData.m_Direction,
        launchParams.m_MinMaxDistance.x,
        rayData.m_MaxDistance,
        0.f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0,
        1,
        0,
        isIntersection);

    //3. If no hit, accumulate result in buffer
    if(isIntersection == 0)
    {

        using namespace WaveFront;

        unsigned int resultIndex =
            static_cast<unsigned int>(LightChannel::NUM_CHANNELS) * rayData.m_PixelIndex +
            static_cast<unsigned int>(rayData.m_OutputChannel);

        launchParams.m_ResultBuffer[resultIndex] += rayData.m_PotentialRadiance;
    }

    return;

}

extern "C"
__device__ __forceinline__ void ReSTIRRayGen()
{
    //Launch as a 1D Array so that idx.x corresponds to the literal ray index.
    unsigned int idx = optixGetLaunchIndex().x;

    //Retrieve the data.
    const RestirShadowRay& rayData = *launchParams.m_ReSTIRShadowRayBatch->GetData(idx);
    const OptixTraversableHandle scene = launchParams.m_TraversableHandle;
    auto reservoirIndex = rayData.index;

    unsigned int intersected = 0;

    optixTrace(
        scene,
        rayData.origin,
        rayData.direction,
        launchParams.m_MinMaxDistance.x, //Prevent self shadowing so offset a little bit.
        rayData.distance,   //Max distance already has a small offset to prevent self-shadowing.
        0.f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0,
        0,
        0,
        intersected  //Pass the reservoir index so that it can be set to 0 when a hit is found.
    );

    if(intersected != 0)
    {
        launchParams.m_Reservoirs[reservoirIndex].weight = 0.f;
    }
}



extern "C"
__global__ void __raygen__WaveFrontRG()
{

	switch (launchParams.m_TraceType)
	{
    case WaveFront::RayType::INTERSECTION_RAY:
        //Primary rays
        IntersectionRaysRayGen();
        break;
    case WaveFront::RayType::SHADOW_RAY:
        //Shadow rays
        ShadowRaysRayGen();
        break;
    case WaveFront::RayType::RESTIR_RAY:
        {
            //ReSTIR Rays.
            ReSTIRRayGen();
        }

        break;
	}

    return;

}


extern "C"
__global__ void __miss__WaveFrontMS()
{

    /*switch (launchParams.m_TraceType)
    {
    case WaveFront::RayType::INTERSECTION_RAY:
        return;
        break;
    case WaveFront::RayType::SHADOW_RAY:
        return;
        break;
    case WaveFront::RayType::RESTIR_RAY:
        return;
        break;
    }*/

    return;

}


extern "C"
__global__ void __anyhit__WaveFrontAH()
{

    switch (launchParams.m_TraceType)
    {
    case WaveFront::RayType::INTERSECTION_RAY:
        break;
    case WaveFront::RayType::SHADOW_RAY:
        {
            optixSetPayload_0(1);
            optixTerminateRay();
        }
        break;
    case WaveFront::RayType::RESTIR_RAY:
        {
            //Any hit is enough.
            optixSetPayload_0(1);
            optixTerminateRay();
        }
        break;
    }

    return;

}


extern "C"
__global__ void __closesthit__WaveFrontCH()
{

    switch (launchParams.m_TraceType)
    {
    case WaveFront::RayType::INTERSECTION_RAY:
        {
            //If closest hit found, return IntersectionData.
            const unsigned int intersectionPtr_Up = optixGetPayload_0();
            const unsigned int intersectionPtr_Low = optixGetPayload_1();
            WaveFront::IntersectionData* intersection = UnpackPointer<WaveFront::IntersectionData>(intersectionPtr_Up, intersectionPtr_Low);

            ////TODO: Try to fit this into 4 floats and one write.
            intersection->m_IntersectionT = optixGetRayTmax();
            intersection->m_Barycentrics = optixGetTriangleBarycentrics();
            intersection->m_PrimitiveIndex = optixGetPrimitiveIndex();
            intersection->m_InstanceId = optixGetInstanceId();

        }    
        break;
    case WaveFront::RayType::SHADOW_RAY:
        break;
    case WaveFront::RayType::RESTIR_RAY:
        //Get the reservoir and set its weight to 0 so that it is no longer considered a valid candidate.
        //reSTIRParams.reservoirs[optixGetAttribute_0()].weight = 0.f;
        //optixTerminateRay();
        break;
    }

    return;

}