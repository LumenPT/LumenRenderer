#include "CppCommon/WaveFrontDataStructs.h"
#include "CppCommon/RenderingUtility.h"
#include "CppCommon/Half4.h"

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

    using namespace WaveFront;
 
    //1. Get ray definition from buffer
    const unsigned idx = optixGetLaunchIndex().x;
    const WaveFront::IntersectionRayData& rayData = *launchParams.m_IntersectionRayBatch->GetData(idx);

    //2. Trace ray: optixTrace()
    WaveFront::IntersectionDataUint4 intersection{};

    unsigned int intersectionPtr_Up = 0;
    unsigned int intersectionPtr_Low = 0;

    PackPointer(&intersection, intersectionPtr_Up, intersectionPtr_Low);

    const OptixTraversableHandle scene = launchParams.m_TraversableHandle;

    //Solid trace
    optixTrace(
        scene,
        rayData.m_Origin,
        rayData.m_Direction,
        launchParams.m_MinMaxDistance.x,
        launchParams.m_MinMaxDistance.y,
        0.f, //Ray Time, can be 0 in our case.
        TraceMaskType::SOLIDS,
        OPTIX_RAY_FLAG_NONE,
        0, //SBT offset for selecting the SBT records to use
        0, //SBT stride for selecting the SBT records to use, multiplied with SBT-GAS index
        0, //Miss SBT index, always use first miss shader.
        intersectionPtr_Up,
        intersectionPtr_Low);

    //3. Store IntersectionData in buffer
    launchParams.m_IntersectionBuffer->data[idx] = intersection.m_Data;


    //TODO: implement volumetric trace and storing of data.
    WaveFront::VolumetricIntersectionData volIntersection{};
    volIntersection.m_RayArrayIndex = idx;
    volIntersection.m_PixelIndex = rayData.m_PixelIndex;

    unsigned int volIntersectionPtr_Up = 0;
    unsigned int volIntersectionPtr_Low = 0;

    PackPointer(&volIntersection, volIntersectionPtr_Up, volIntersectionPtr_Low);

    //Volumetric trace
    optixTrace(
        scene,
        rayData.m_Origin,
        rayData.m_Direction,
        launchParams.m_MinMaxDistance.x,
        min(launchParams.m_MinMaxDistance.y, intersection.m_Data.m_IntersectionT),
        0.f,
        TraceMaskType::VOLUMES,
        OPTIX_RAY_FLAG_NONE,
        0,
        0,
        0,
        volIntersectionPtr_Up,
        volIntersectionPtr_Low);

    launchParams.m_VolumetricIntersectionBuffer->Add(&volIntersection);

    return;

}

extern "C"
__device__ __forceinline__ void ShadowRaysRayGen()
{

    using namespace WaveFront;

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
        TraceMaskType::SOLIDS,
        OPTIX_RAY_FLAG_NONE,
        0,
        1,
        0,
        isIntersection);

	//TODO: volumetric shadows

    //3. If no hit, accumulate result in buffer
    if(isIntersection == 0)
    {
        using namespace WaveFront;

        
        half4Ushort4 color{ make_ushort4(0, 0 , 0, 0) };

        surf2Dread<ushort4>(
            &color.m_Ushort4,
            launchParams.m_OutputChannels[static_cast<unsigned int>(rayData.m_OutputChannel)],
            rayData.m_PixelIndex.m_X * sizeof(ushort4),
            rayData.m_PixelIndex.m_Y,
            cudaBoundaryModeTrap);

        //color += half4(rayData.m_PotentialRadiance, 0.f);

        half4 radiance{ rayData.m_PotentialRadiance, 0.f };

        color.m_Half4.m_Elements[0].x = color.m_Half4.m_Elements[0].x + radiance.m_Elements[0].x;
        color.m_Half4.m_Elements[0].y = color.m_Half4.m_Elements[0].y + radiance.m_Elements[0].y;
        color.m_Half4.m_Elements[1].x = color.m_Half4.m_Elements[1].x + radiance.m_Elements[1].x;
        color.m_Half4.m_Elements[1].y = color.m_Half4.m_Elements[1].y + radiance.m_Elements[1].y;

        surf2Dwrite<ushort4>(
            color.m_Ushort4,
            launchParams.m_OutputChannels[static_cast<unsigned int>(rayData.m_OutputChannel)],
            rayData.m_PixelIndex.m_X * sizeof(ushort4),
            rayData.m_PixelIndex.m_Y,
            cudaBoundaryModeTrap);

    }

    return;

}

extern "C"
__device__ __forceinline__ void ReSTIRRayGen()
{

    using namespace WaveFront;

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
        TraceMaskType::SOLIDS,
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

    using namespace WaveFront;

	switch (launchParams.m_TraceType)
	{
    case RayType::INTERSECTION_RAY:
        //Primary rays
        IntersectionRaysRayGen();
        break;
    case RayType::SHADOW_RAY:
        //Shadow rays
        ShadowRaysRayGen();
        break;
    case RayType::RESTIR_RAY:
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

    //using namespace WaveFront;

    /*switch (launchParams.m_TraceType)
    {
    case RayType::INTERSECTION_RAY:
        return;
        break;
    case RayType::SHADOW_RAY:
        return;
        break;
    case RayType::RESTIR_RAY:
        return;
        break;
    }*/

    return;

}


extern "C"
__global__ void __anyhit__WaveFrontAH()
{

    using namespace WaveFront;

    switch (launchParams.m_TraceType)
    {
    case RayType::INTERSECTION_RAY:
        break;
    case RayType::SHADOW_RAY:
        {
            optixSetPayload_0(1);
            optixTerminateRay();
        }
        break;
    case RayType::RESTIR_RAY:
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

    using namespace WaveFront;

    switch (launchParams.m_TraceType)
    {
    case RayType::INTERSECTION_RAY:
        {
            //If closest hit found, return IntersectionData.
            const unsigned int intersectionPtr_Up = optixGetPayload_0();
            const unsigned int intersectionPtr_Low = optixGetPayload_1();
            WaveFront::IntersectionDataUint4* intersection = UnpackPointer<WaveFront::IntersectionDataUint4>(intersectionPtr_Up, intersectionPtr_Low);

    		//Pack the data into a uint 4 and locally.
            WaveFront::IntersectionDataUint4 localData;
            const auto barycentrics = optixGetTriangleBarycentrics();
            localData.m_Data.m_IntersectionT = optixGetRayTmax();
            localData.m_Data.m_Barycentrics.x = barycentrics.x;
            localData.m_Data.m_Barycentrics.y = barycentrics.y;
            localData.m_Data.m_PrimitiveIndex = optixGetPrimitiveIndex();
            localData.m_Data.m_InstanceId = optixGetInstanceId();

    		//Access external memory once by copying over the 128 bits.
            intersection->m_DataAsUint4 = localData.m_DataAsUint4;
        }    
        break;
    case RayType::SHADOW_RAY:
        break;
    case RayType::RESTIR_RAY:
        //Get the reservoir and set its weight to 0 so that it is no longer considered a valid candidate.
        //reSTIRParams.reservoirs[optixGetAttribute_0()].weight = 0.f;
        //optixTerminateRay();
        break;
    }

    return;

}