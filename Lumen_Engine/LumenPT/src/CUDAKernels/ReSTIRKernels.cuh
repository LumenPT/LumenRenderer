#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>

#include "../../vendor/Include/Cuda/cuda/helpers.h"
#include "../Shaders/CppCommon/ReSTIRData.h"

namespace WaveFront {
    struct RayData;
    struct IntersectionData;
}

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
__host__ void PickPrimarySamples(const WaveFront::RayData* const a_RayData, const WaveFront::IntersectionData* const a_IntersectionData, const LightBagEntry* const a_LightBags, Reservoir* a_Reservoirs, const ReSTIRSettings& a_Settings);

__global__ void PickPrimarySamplesInternal(const WaveFront::RayData* const a_RayData, const WaveFront::IntersectionData* const a_IntersectionData, const LightBagEntry* const a_LightBags, Reservoir* a_Reservoirs, const unsigned a_NumPrimarySamples, const unsigned a_NumReservoirs, const unsigned a_NumLightsPerBag, const unsigned a_NumLightBags);

/*
 * Generate shadow rays for the given reservoirs.
 * These shadow rays are then resolved with Optix, and reservoirs weights are updated to reflect mutual visibility.
 */
__host__ void VisibilityPass(Reservoir* a_Reservoirs, unsigned a_NumReservoirs);

/*
 * Resample spatial neighbours with an intermediate output buffer.
 * Combine reservoirs.
 */
__host__ void SpatialNeighbourSampling(Reservoir* a_Reservoirs, Reservoir* a_ReservoirSwapBuffer, const ReSTIRSettings& a_Settings);

/*
 * Resample temporal neighbours and combine reservoirs.
 */
__host__ void TemporalNeighbourSampling(Reservoir* a_Reservoirs, Reservoir* a_TemporalReservoirs, const ReSTIRSettings& a_Settings);