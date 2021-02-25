#include <cstdio>
#include <cuda/helpers.h>

#include "../../vendor/Include/sutil/vec_math.h"
#include "Optix/optix.h"

#include "nanovdb/NanoVDB.h"
#include <nanovdb/util/Primitives.h>
#include <nanovdb/util/IO.h>
#include <nanovdb/util/Ray.h>
#include <nanovdb/util/HDDA.h>

extern "C"
__global__ void __closesthit__VolumetricHitShader()
{
    //const nanovdb::FloatGrid* tempGrid = nullptr;

    //auto& grid = *tempGrid;

    optixSetPayload_0(float_as_int(0.0f));
    optixSetPayload_1(float_as_int(0.0f));
    optixSetPayload_2(float_as_int(1.0f));
    optixSetPayload_3(1);
	
    printf("test");

    return;
}

extern "C"
__global__ void __anyhit__VolumetricHitShader()
{
    printf("test");
    return;
}

extern "C"
__global__ void __intersection__VolumetricHitShader()
{
    printf("test");
	
    make_float3(0.f, 0.f, 0.f);

    optixReportIntersection(0.f, 0);
	
    return;
}