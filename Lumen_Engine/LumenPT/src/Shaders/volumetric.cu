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

	float entry = int_as_float(optixGetAttribute_0());
	float exit = int_as_float(optixGetAttribute_1());
	exit = min(exit, int_as_float(optixGetPayload_4()));
	
	const int MAX_STEPS = 1000;
	const float STEP_SIZE = 1.0f;
	const float HARDCODED_DENSITY_PER_STEP = 0.005f;
	const float VOLUME_COLOR_R = 1.0f;
	const float VOLUME_COLOR_G = 1.0f;
	const float VOLUME_COLOR_B = 1.0f;
	
	float distance = exit - entry;
	int necessarySteps = int(distance / STEP_SIZE);
	int nSteps = min(necessarySteps, MAX_STEPS);
	float accumulatedDensity = 0.0f;
	
	for (int i = 0; i < nSteps && accumulatedDensity < 1.0f; i++)
	{
		//Volumetric sampling code goes here
		accumulatedDensity += HARDCODED_DENSITY_PER_STEP;
	}

	float r, g, b;
	r = int_as_float(optixGetPayload_0());
	g = int_as_float(optixGetPayload_1());
	b = int_as_float(optixGetPayload_2());

	
	r = VOLUME_COLOR_R * accumulatedDensity + r * (1 - accumulatedDensity);
	g = VOLUME_COLOR_G * accumulatedDensity + g * (1 - accumulatedDensity);
	b = VOLUME_COLOR_B * accumulatedDensity + b * (1 - accumulatedDensity);

	optixSetPayload_0(float_as_int(r));
	optixSetPayload_1(float_as_int(g));
	optixSetPayload_2(float_as_int(b));
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
    //{
	//	auto bbox = grid.tree().bbox();
	//
	//
    //    if (iRay.clip(bbox) == true)
    //    {
    //        optixReportIntersection(1.f, 0,
	//			float_as_int();
    //    }
	//
    //}

	auto bbox = grid.worldBBox();
	float t0;	//volume entry point
	float t1;	//volume exit point
	if (wRay.intersects(bbox, t0, t1))
	{
		optixReportIntersection(t0, 0,
			float_as_int(t0),
			float_as_int(t1));
	}

	////temp code to visualize voxel outlines
    //auto acc = grid.tree().getAccessor();
    //float iT0;
    //nanovdb::Coord ijk;
    //float v;
	//
	////calculate voxel grid dimensions
	//auto minGrid = grid.indexBBox().min();
	//auto maxGrid = grid.indexBBox().max();
	//float gridWidth = maxGrid.x() - minGrid.x();
	//float gridHeight = maxGrid.y() - minGrid.y();
	//float gridDepth = maxGrid.z() - minGrid.z();
	//
	//
    //if (nanovdb::ZeroCrossing(iRay, acc, ijk, v, iT0))
    //{
    //    optixReportIntersection(1.f, 0,
	//		float_as_int(ijk.x() / gridWidth + 0.5f),
	//		float_as_int(ijk.y() / gridHeight + 0.5f),
	//		float_as_int(ijk.z() / gridDepth + 0.5f));
	//	//optixReportIntersection(1.0f, 0);
    //}


    //nanovdb::Ray<float> iRay = wRay.worldToIndexF(*tempGrid);

    make_float3(0.f, 0.f, 0.f);

    //optixReportIntersection(1.f, 0);

    return;
}