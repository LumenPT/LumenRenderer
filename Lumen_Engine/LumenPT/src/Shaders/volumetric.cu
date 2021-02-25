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
    const nanovdb::FloatGrid* tempGrid = nullptr;

    auto& grid = *tempGrid;

    printf("%s \n", grid.gridName());
	
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
    return;
}