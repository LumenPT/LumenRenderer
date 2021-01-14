#include "CppCommon/WaveFrontDataStructs.h"
#include "CppCommon/RenderingUtility.h"

#include <cuda/helpers.h>
#include <Optix/optix.h>
#include "../../vendor/Include/sutil/vec_math.h"

extern "C"
{
    __constant__ WaveFront::CommonOptixLaunchParameters commonParameters;
    __constant__ WaveFront::ResolveRaysLaunchParameters resolveRaysParams;

}

extern "C"
{

    __global__ void __raygen__ResolveRaysRayGen()
    {

        //Data available: resolveRaysParams and commonParameters

        //For each pixel:
        //1. Get ray definition from buffer
        //2. Trace ray: optixTrace()
        //3. Store IntersectionData in buffer
    }

    __global__ void __closesthit__ResolveRaysClosestHit()
    {

        //Data available: resolveRaysParams and commonParameters
        //If closest hit found, return IntersectionData.

    }

}


extern "C"
{

    __constant__ WaveFront::ResolveShadowRaysLaunchParameters resolveShadowRaysParams;

}

extern "C"
{

    __global__ void __raygen__ResolveShadowRaysRayGen()
    {

        //Data available: resolveShadowRaysParams and commonParameters

        //For each pixel:
        //For depth (num rays per pixel)
        //1. Get shadow ray definition from buffer
        //2. Trace ray: optixTrace()
        //3. If no hit, accumulate result in buffer;

    }

    __global__ void __anyhit__ResolveShadowRaysAnyHit()
    {

        //Data available: resolveShadowRaysParams and commonParameters
        //If any hit found before max distance, return true;

    }

}