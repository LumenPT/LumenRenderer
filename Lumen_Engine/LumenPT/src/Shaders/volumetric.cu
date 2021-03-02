#include <cstdio>
#include <cuda/helpers.h>

#include "../../vendor/Include/sutil/vec_math.h"
#include "Optix/optix.h"

#include "nanovdb/NanoVDB.h"
#include <nanovdb/util/Primitives.h>
#include <nanovdb/util/IO.h>
#include <nanovdb/util/Ray.h>
#include <nanovdb/util/HDDA.h>

#include "VolumeStructs.h"

extern "C"
__global__ void __closesthit__VolumetricHitShader()
{
    //const nanovdb::FloatGrid* tempGrid = nullptr;

    //auto& grid = *tempGrid;

    optixSetPayload_0(float_as_int(1.0f));
    optixSetPayload_1(float_as_int(0.0f));
    optixSetPayload_2(float_as_int(1.0f));
    optixSetPayload_3(1);


    return;
}

extern "C"
__global__ void __anyhit__VolumetricHitShader()
{
    return;
}

extern "C"
__global__ void __intersection__VolumetricHitShader()
{
    DeviceVolume* volume = reinterpret_cast<DeviceVolume*>(optixGetSbtDataPointer());;

    const nanovdb::FloatGrid* pGrid = volume->m_Grid;

    auto& grid = *pGrid;

    //printf("%s \n", grid.gridName());

    const float3 ray_orig = optixGetObjectRayOrigin();
    const float3 ray_dir = optixGetObjectRayDirection();
    const float  ray_tmin = optixGetRayTmin();
    const float  ray_tmax = optixGetRayTmax();

    nanovdb::Ray<float> wRay(
        nanovdb::Vec3<float>(ray_orig.x, ray_orig.y, ray_orig.z),
        nanovdb::Vec3<float>(ray_dir.x, ray_dir.y, ray_dir.z),
        ray_tmin,
        ray_tmax
    );

    nanovdb::Ray<float> iRay = wRay.worldToIndexF(grid);

    //temp test code, to delete
    {
        auto bbox = grid.tree().bbox();

        if (iRay.clip(bbox) == true)
        {
            //optixReportIntersection(1.f, 0);
        }

    }

    auto acc = grid.tree().getAccessor();
    float iT0;
    nanovdb::Coord ijk;
    float v;

    if (nanovdb::ZeroCrossing(iRay, acc, ijk, v, iT0))
    {
        optixReportIntersection(1.f, 0);
    }


    //nanovdb::Ray<float> iRay = wRay.worldToIndexF(*tempGrid);

    make_float3(0.f, 0.f, 0.f);

    //optixReportIntersection(1.f, 0);

    return;
}