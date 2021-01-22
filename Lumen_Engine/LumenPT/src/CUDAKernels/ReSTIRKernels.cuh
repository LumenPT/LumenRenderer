#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>

#include "../../vendor/Include/Cuda/cuda/helpers.h"
#include "../Shaders/CppCommon/ReSTIRData.h"

__host__ void ResetReservoirs(int a_NumReservoirs, Reservoir* a_ReservoirPointer);

__global__ void ResetReservoirInternal(int a_NumReservoirs, Reservoir* a_ReservoirPointer);

__host__ void FillCDF(CDF* a_Cdf, TriangleLight* a_Lights, unsigned a_LightCount);

__global__ void ResetCDF(CDF* a_Cdf);

__global__ void FillCDFInternal(CDF* a_Cdf, TriangleLight* a_Lights, unsigned a_LightCount);

__host__ void FillLightBags(unsigned a_NumLightBags, CDF* a_Cdf, LightBagEntry* a_LightBagPtr, TriangleLight* a_Lights);

__global__ void FillLightBagsInternal(unsigned a_NumLightBags, CDF* a_Cdf, LightBagEntry* a_LightBagPtr, TriangleLight* a_Lights);

/*
 * Prick primary lights and apply reservoir sampling.
 */
__host__ void PickPrimarySamples(LightBagEntry* a_LightBags, Reservoir* a_Reservoirs, const int2& a_Dimensions, unsigned a_PixelGridSize);

__global__ void PickPrimarySamplesInternal(LightBagEntry* a_LightBags, Reservoir* a_Reservoirs, const int2& a_Dimensions, unsigned a_PixelGridSize);