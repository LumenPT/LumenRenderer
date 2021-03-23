#include "CppCommon/WaveFrontDataStructs.h"
#include "../../vendor/Include/Cuda/cuda/helpers.h"
#include "../../vendor/Include/sutil/vec_math.h"
#include "../../vendor/Include/Optix/optix.h"
#include "../../vendor/Include/Optix/optix_device.h"

#include <cstdio>

extern "C"
{

    __constant__ WaveFront::OptixLaunchParameters launchParams;

}



extern "C"
__global__ void __closesthit__Volumetric()
{

    return;
}

extern "C"
__global__ void __anyhit__Volumetric()
{

    return;
}

extern "C"
__global__ void __intersection__Volumetric()
{
    optixReportIntersection(1.f, 0, 
        0, 0);

    return;
}