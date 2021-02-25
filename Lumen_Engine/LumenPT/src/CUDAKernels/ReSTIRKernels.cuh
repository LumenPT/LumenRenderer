#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>

#include "../../vendor/Include/Cuda/cuda/helpers.h"
#include "../Shaders/CppCommon/ReSTIRData.h"
#include "../Framework/MemoryBuffer.h"

/*
 * Macros used to access elements at a certain index.
 */
#define RESERVOIR_INDEX(INTERSECTION_INDEX, DEPTH, MAX_DEPTH) (((INTERSECTION_INDEX) * (MAX_DEPTH)) + (DEPTH))

class MemoryBuffer;

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

__global__ void PickPrimarySamplesInternal(const WaveFront::RayData* const a_RayData, const WaveFront::IntersectionData* const a_IntersectionData, const LightBagEntry* const a_LightBags, Reservoir* a_Reservoirs, const ReSTIRSettings& a_Settings);

/*
 * Generate shadow rays for the given reservoirs.
 * These shadow rays are then resolved with Optix, and reservoirs weights are updated to reflect mutual visibility.
 */
__host__ void VisibilityPass(MemoryBuffer* a_AtomicCounter, Reservoir* a_Reservoirs, const WaveFront::IntersectionData* a_IntersectionData, const WaveFront::RayData* const a_RayData, unsigned a_NumReservoirsPerPixel, const std::uint32_t a_NumPixels, RestirShadowRay* a_ShadowRays);

//Generate a shadow ray based on the thread ID.
__global__ void GenerateShadowRay(int* a_AtomicCounter, const std::uint32_t a_NumPixels, std::uint32_t a_NumReservoirsPerPixel, const WaveFront::RayData* const a_RayData, Reservoir* a_Reservoirs, const WaveFront::IntersectionData* a_IntersectionData,  RestirShadowRay* a_ShadowRays);

/*
 * Resample spatial neighbours with an intermediate output buffer.
 * Combine reservoirs.
 */
__host__ void SpatialNeighbourSampling(Reservoir* a_Reservoirs, Reservoir* a_ReservoirSwapBuffer, const ReSTIRSettings& a_Settings);

/*
 * Resample temporal neighbours and combine reservoirs.
 */
__host__ void TemporalNeighbourSampling(
    Reservoir* a_CurrentReservoirs,
    Reservoir* a_PreviousReservoirs,
    const WaveFront::IntersectionData* a_CurrentIntersectionData,
    const WaveFront::IntersectionData* a_PreviousIntersectionData,
    const WaveFront::RayData* const a_CurrentRayData,
    const WaveFront::RayData* const a_PreviousRayData,
    const ReSTIRSettings& a_Settings);

__global__ void CombineTemporalSamplesInternal(
    int a_NumPixels,
    Reservoir* a_CurrentReservoirs,
    Reservoir* a_PreviousReservoirs,
    const WaveFront::IntersectionData* a_CurrentIntersectionData,
    const WaveFront::IntersectionData* a_PreviousIntersectionData,
    const WaveFront::RayData* const a_CurrentRayData,
    const WaveFront::RayData* const a_PreviousRayData,
    const bool a_Biased);

/*
 * Combine multiple reservoirs unbiased.
 * Stores the result in the reservoirs with the given pixel index.
 *
 * a_Count indicates the amount of pixels to process in a_ToCombine.
 * This runs for every reservoir at a_PixelIndex.
 */
__device__ void CombineUnbiased(int a_PixelIndex, int a_Count, int a_MaxReservoirDepth, Reservoir** a_Reservoirs, PixelData** a_ToCombine);

/*
 * Combine multiple reservoirs biased.
 * Stores the result in the reservoirs with the given pixel index.
 *
 * a_Count indicates the amount of indices to process in a_ToCombineIndices.
 * This runs for every reservoir depth.
 */
__device__ void CombineBiased(int a_PixelIndex, int a_Count, int a_MaxReservoirDepth, Reservoir** a_Reservoirs, PixelData** a_ToCombine);

/*
 * Resample an old light sample.
 * This takes the given input, and resamples it against the given pixel data.
 * The output is then stored in a_Output.
 *
 */
__device__ void Resample(LightSample* a_Input, const PixelData* a_PixelData, LightSample* a_Output);