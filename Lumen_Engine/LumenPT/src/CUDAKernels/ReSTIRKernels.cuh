#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>

#include "../../vendor/Include/Cuda/cuda/helpers.h"
#include "../Shaders/CppCommon/ReSTIRData.h"
#include "../Framework/MemoryBuffer.h"
#include "../Shaders/CppCommon/WaveFrontDataStructs.h"
#include "../CUDAKernels/RandomUtilities.cuh"

/*
 * Macros used to access elements at a certain index.
 */
#define RESERVOIR_INDEX(INTERSECTION_INDEX, DEPTH, MAX_DEPTH) (((INTERSECTION_INDEX) * (MAX_DEPTH)) + (DEPTH))
#define PIXEL_INDEX(X, Y, WIDTH) (((Y) * (WIDTH) + (X)))

class MemoryBuffer;

namespace WaveFront {
    struct RayData;
    struct IntersectionData;
    struct TriangleLight;
}

__host__ void ResetReservoirs(int a_NumReservoirs, Reservoir* a_ReservoirPointer);

__global__ void ResetReservoirInternal(int a_NumReservoirs, Reservoir* a_ReservoirPointer);

__host__ void FillCDF(CDF* a_Cdf, WaveFront::TriangleLight* a_Lights, unsigned a_LightCount);

__global__ void ResetCDF(CDF* a_Cdf);

__global__ void FillCDFInternal(CDF* a_Cdf, WaveFront::TriangleLight* a_Lights, unsigned a_LightCount);

__host__ void FillLightBags(unsigned a_NumLightBags, CDF* a_Cdf, LightBagEntry* a_LightBagPtr, WaveFront::TriangleLight* a_Lights, const std::uint32_t a_Seed);

__global__ void FillLightBagsInternal(unsigned a_NumLightBags, CDF* a_Cdf, LightBagEntry* a_LightBagPtr, WaveFront::TriangleLight* a_Lights, const std::uint32_t a_Seed);

/*
 * Prick primary lights and apply reservoir sampling.
 */
__host__ void PickPrimarySamples(const LightBagEntry* const a_LightBags, Reservoir* a_Reservoirs, const ReSTIRSettings& a_Settings, WaveFront::SurfaceData* a_PixelData, const std::uint32_t a_Seed);

__global__ void PickPrimarySamplesInternal(const LightBagEntry* const a_LightBags, Reservoir* a_Reservoirs, const ReSTIRSettings& a_Settings, WaveFront::SurfaceData* a_PixelData, const std::uint32_t a_Seed);

/*
 * Generate shadow rays for the given reservoirs.
 * These shadow rays are then resolved with Optix, and reservoirs weights are updated to reflect mutual visibility.
 *
 * The amount of shadow rays that was generated will be returned as an int.
 */
__host__ int GenerateReSTIRShadowRays(MemoryBuffer* a_AtomicCounter, Reservoir* a_Reservoirs, RestirShadowRay* a_ShadowRays, WaveFront::SurfaceData* a_PixelData);

//Generate a shadow ray based on the thread ID.
__global__ void GenerateShadowRay(int* a_AtomicCounter, Reservoir* a_Reservoirs, RestirShadowRay* a_ShadowRays, WaveFront::SurfaceData* a_PixelData);

/*
 * Resample spatial neighbours with an intermediate output buffer.
 * Combine reservoirs.
 */
__host__ void SpatialNeighbourSampling(
    Reservoir* a_Reservoirs,
    Reservoir* a_SwapBuffer,
    WaveFront::SurfaceData* a_PixelData,
    const std::uint32_t a_Seed
);

__global__ void SpatialNeighbourSamplingInternal(
    Reservoir* a_Reservoirs,
    Reservoir* a_SwapBuffer,
    WaveFront::SurfaceData* a_PixelData,
    const std::uint32_t a_Seed
);

/*
 * Resample temporal neighbours and combine reservoirs.
 */
__host__ void TemporalNeighbourSampling(
    Reservoir* a_CurrentReservoirs,
    Reservoir* a_PreviousReservoirs,
    WaveFront::SurfaceData* a_CurrentPixelData,
    WaveFront::SurfaceData* a_PreviousPixelData,
    const std::uint32_t a_Seed
);

__global__ void CombineTemporalSamplesInternal(
    Reservoir* a_CurrentReservoirs,
    Reservoir* a_PreviousReservoirs,
    WaveFront::SurfaceData* a_CurrentPixelData,
    WaveFront::SurfaceData* a_PreviousPixelData,
    const std::uint32_t a_Seed
);

/*
 * Combine multiple reservoirs unbiased.
 * Stores the result in the reservoirs with the given pixel index.
 *
 * a_Count indicates the amount of pixels to process in a_ToCombine.
 * This runs for every reservoir at a_PixelIndex.
 */
__device__ void CombineUnbiased(int a_PixelIndex, int a_Count, Reservoir** a_Reservoirs, WaveFront::SurfaceData** a_ToCombine, const std::uint32_t a_Seed);

/*
 * Combine multiple reservoirs biased.
 * Stores the result in the reservoirs with the given pixel index.
 *
 * a_Count indicates the amount of indices to process in a_ToCombineIndices.
 * This runs for every reservoir depth.
 */
__device__ void CombineBiased(int a_PixelIndex, int a_Count, Reservoir** a_Reservoirs, WaveFront::SurfaceData** a_ToCombine, const std::uint32_t a_Seed);

/*
 * Resample an old light sample.
 * This takes the given input, and resamples it against the given pixel data.
 * The output is then stored in a_Output.
 *
 */
__device__ void Resample(LightSample* a_Input, const WaveFront::SurfaceData* a_PixelData, LightSample* a_Output);

/*
 * Generate shadow rays from the reservoirs after ReSTIR runs.
 * Add the shadow rays with scaled-by-weight payload to the wavefront buffer.
 */
__host__ void GenerateWaveFrontShadowRays(Reservoir* a_Reservoirs, WaveFront::SurfaceData* a_PixelData, WaveFront::AtomicBuffer<WaveFront::ShadowRayData>* a_AtomicBuffer);

__global__ void GenerateWaveFrontShadowRaysInternal(Reservoir* a_Reservoirs, WaveFront::SurfaceData* a_PixelData, WaveFront::AtomicBuffer<WaveFront::ShadowRayData>* a_AtomicBuffer);