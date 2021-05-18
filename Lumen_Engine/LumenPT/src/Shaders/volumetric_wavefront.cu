#include "CppCommon/WaveFrontDataStructs.h"
#include "../../vendor/Include/Cuda/cuda/helpers.h"
#include "../../vendor/Include/sutil/vec_math.h"
#include "../../vendor/Include/Optix/optix.h"
#include "../../vendor/Include/Optix/optix_device.h"

#include <cstdio>
#include "CppCommon/VolumeStructs.h"
#include "../../vendor/openvdb/nanovdb/nanovdb/util/GridHandle.h"
#include "../../vendor/openvdb/nanovdb/nanovdb/util/Ray.h"
#include "../../vendor/openvdb/nanovdb/nanovdb/util/HDDA.h"

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

extern "C"
__global__ void __closesthit__Volumetric()
{

	WaveFront::VolumetricIntersectionData* intersection = UnpackPointer<WaveFront::VolumetricIntersectionData>(optixGetPayload_0(), optixGetPayload_1());

	if (optixGetPayload_0() > 0)
	{
		DeviceVolume* volume = launchParams.m_SceneData->GetTableEntry<DeviceVolume>(optixGetInstanceId());
		const nanovdb::FloatGrid* pGrid = volume->m_Grid;

		intersection->m_EntryT = uint_as_float(optixGetAttribute_0());
		intersection->m_ExitT = uint_as_float(optixGetAttribute_1());
		intersection->m_VolumeGrid = pGrid;
	}

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

    const unsigned int instanceId = optixGetInstanceId();

    DeviceVolume* volume = launchParams.m_SceneData->GetTableEntry<DeviceVolume>(instanceId);
    const nanovdb::FloatGrid* pGrid = volume->m_Grid;

    auto& grid = *pGrid;
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

    {	
		auto bbox = grid.worldBBox();
		float t0;	//volume entry point
		float t1;	//volume exit point
		if (wRay.intersects(bbox, t0, t1) && t0 < optixGetRayTmax())
		{
			optixReportIntersection(t0, 0,
				float_as_int(t0),
				float_as_int(t1));
		}
		
    }

    return;
}