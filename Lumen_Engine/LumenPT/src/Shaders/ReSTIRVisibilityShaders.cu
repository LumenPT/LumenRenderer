#include "CppCommon/WaveFrontDataStructs.h"
#include "CppCommon/RenderingUtility.h"

#include "../../vendor/Include/Cuda/cuda/helpers.h"
#include "../../vendor/Include/Optix/optix_device.h"

//extern "C"
//{
//    __constant__ ReSTIROptixParameters reSTIRParams;
//}
//
//extern "C"
//__global__ void __raygen__rg()
//{
//    //Launch as a 1D Array so that idx.x corresponds to the literal ray index.
//    const uint3 idx = optixGetLaunchIndex();
//
//    //Retrieve the data.
//    const RestirShadowRay& rayData = reSTIRParams.shadowRays[idx.x];
//    const OptixTraversableHandle scene = reSTIRParams.optixSceneHandle;
//    auto reservoirIndex = rayData.index;
//
//    optixTrace(
//        scene,
//        rayData.origin,
//        rayData.direction,
//        0.005f, //Prevent self shadowing so offset a little bit.
//        rayData.distance,   //Max distance already has a small offset to prevent self-shadowing.
//        0.f,
//        OptixVisibilityMask(255),
//        OPTIX_RAY_FLAG_NONE,
//        0,  //TODO
//        1,  //TODO
//        0,  //TODO
//        reservoirIndex  //Pass the reservoir index so that it can be set to 0 when a hit is found.
//    );
//}
//
//extern "C"
//__global__ void __anyhit__anyhit()
//{
//    //Get the reservoir and set its weight to 0 so that it is no longer considered a valid candidate.
//    reSTIRParams.reservoirs[optixGetAttribute_0()].weight = 0.f;
//    optixTerminateRay();
//}
//
//extern "C"
//__global__ void __miss__miss()
//{
//    //Nothing needs to be done on miss.
//}