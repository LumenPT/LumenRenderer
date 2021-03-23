#include <cstdio>
#include <cuda/helpers.h>

#include "Optix/optix.h"


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