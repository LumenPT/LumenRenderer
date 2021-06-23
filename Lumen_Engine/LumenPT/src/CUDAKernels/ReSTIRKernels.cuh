#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>

#include "../../vendor/Include/Cuda/cuda/helpers.h"
#include "../Shaders/CppCommon/ReSTIRData.h"
#include "../Framework/MemoryBuffer.h"
#include "../Shaders/CppCommon/WaveFrontDataStructs.h"
#include "../CUDAKernels/RandomUtilities.cuh"
#include <thrust/sort.h>

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


struct TriangleLightComparator
{
    __host__ __device__ bool operator()(const WaveFront::TriangleLight& lhs, const WaveFront::TriangleLight& rhs) const
	{
        return ((lhs.radiance.x + lhs.radiance.y + lhs.radiance.z) / 3.f) < ((rhs.radiance.x + rhs.radiance.y + rhs.radiance.z) / 3.f);
    }
};

struct CdfComparator
{
    __host__ __device__ bool operator()(const WaveFront::TriangleLight& lhs, const WaveFront::TriangleLight& rhs) const
    {
        return ((lhs.radiance.x + lhs.radiance.y + lhs.radiance.z) / 3.f) + ((rhs.radiance.x + rhs.radiance.y + rhs.radiance.z) / 3.f);
    }
};

__host__ void ResetReservoirs(
    int a_NumReservoirs, 
    Reservoir* a_ReservoirPointer);

__global__ void ResetReservoirInternal(
    int a_NumReservoirs, 
    Reservoir* a_ReservoirPointer);

__host__ void FillCDF(
    CDF* a_Cdf, 
    float* a_CdfTreeBuffer, 
    WaveFront::AtomicBuffer<WaveFront::TriangleLight>* a_Lights, 
    unsigned a_LightCount);

__global__ void ResetCDF(CDF* a_Cdf);

/*
 * Set the CDF sum and size based on the tree.
 */
__global__ void SetCDFSize(CDF* a_Cdf, unsigned a_NumLights);

/*
 * Function used to visualize a CDF and tree to ensure that the values are as expected.
 */
__global__ void DebugPrintCdf(CDF* a_Cdf, float* a_CDFTree);

/*
 * Build a balanced binary tree of all lights bottom up.
 * NumParentNodes contains the amount of nodes at the depth above this depth as a base of 2.
 * It is equal to half the amount of nodes at the current depth.
 */
__global__ void BuildCDFTree(
    float* a_CdfTreeBuffer, 
    unsigned a_NumParentNodes, 
    unsigned a_ArrayOffset);

/*
 * Calculate the weights for each light and output to the CDF tree.
 */
__global__ void CalculateLightWeights(
    float* a_CdfTreeBuffer, 
    const WaveFront::AtomicBuffer<WaveFront::TriangleLight>* a_Lights, 
    unsigned a_LightCount, 
    unsigned a_NumLeafNodes);

/*
 * Calculate light weights and directly put them in the CDF.
 */
__global__ void CalculateLightWeightsInCDF(
    CDF* a_Cdf, 
    const WaveFront::AtomicBuffer<WaveFront::TriangleLight>* a_Lights, 
    unsigned a_LightCount);


/*
 * Traverse the binary tree to fill the CDF with the sums of the light buffer.
 */
__global__ void FillCDFParallel(
    CDF* a_Cdf, 
    float* a_CdfTreeBuffer, 
    const WaveFront::AtomicBuffer<WaveFront::TriangleLight>* a_Lights, 
    unsigned a_LightCount, 
    unsigned a_TreeDepth, 
    unsigned a_LeafNodes);

/*
 * Use a single thread to fill the CDF. Very inefficient.
 */
__global__ void FillCDFInternalSingleThread(
    CDF* a_Cdf, 
    const WaveFront::AtomicBuffer<WaveFront::TriangleLight>* a_Lights, 
    unsigned a_LightCount);


__host__ void FillLightBags(
    unsigned a_NumLightBags, 
    unsigned a_NumLightsPerBag, 
    CDF* a_Cdf, 
    LightBagEntry* a_LightBagPtr, 
    const WaveFront::AtomicBuffer<WaveFront::TriangleLight>* a_Lights, 
    const std::uint32_t a_Seed);

__global__ void FillLightBagsInternal(
    unsigned a_NumLightBags, 
    unsigned a_NumLightsPerBag, 
    CDF* a_Cdf, 
    LightBagEntry* a_LightBagPtr, 
    const WaveFront::AtomicBuffer<WaveFront::TriangleLight>* a_Lights, 
    const std::uint32_t a_Seed);

/*
 * Prick primary lights and apply reservoir sampling.
 */
__host__ void PickPrimarySamples(
    const LightBagEntry* const a_LightBags, 
    Reservoir* a_Reservoirs, 
    const ReSTIRSettings& a_Settings, 
    const WaveFront::SurfaceData* const a_PixelData, 
    const std::uint32_t a_Seed);

__global__ void PickPrimarySamplesInternal(
    const LightBagEntry* const a_LightBags, 
    Reservoir* a_Reservoirs, 
    unsigned a_NumPrimarySamples, 
    unsigned a_NumReservoirs,
    unsigned a_NumLightBags, 
    unsigned a_NumLightsPerBag, 
    const WaveFront::SurfaceData * const a_PixelData, 
    const std::uint32_t a_Seed);

/*
 * Generate shadow rays for the given reservoirs.
 * These shadow rays are then resolved with Optix, and reservoirs weights are updated to reflect mutual visibility.
 * BufferOffset provides the offset into the reservoir buffer at which to write.
 *
 *
 * The amount of shadow rays that was generated will be returned as an int.
 */
__host__ unsigned int GenerateReSTIRShadowRays(
    MemoryBuffer* a_AtomicBuffer, 
    Reservoir* a_Reservoirs, 
    const WaveFront::SurfaceData* a_PixelData, 
    unsigned a_NumReservoirs);

//Generate a shadow ray based on the thread ID.
__global__ void GenerateShadowRay(
    WaveFront::AtomicBuffer<RestirShadowRay>* a_AtomicBuffer, 
    Reservoir* a_Reservoirs, 
    const  WaveFront::SurfaceData* a_PixelData, 
    unsigned a_NumReservoirs);


/*
 * Use reservoirs to write to output buffers.
 * Width and height use the amount of reservoirs per pixel to determine which reservoir contributes to each pixel.
 */
__host__ void Shade(
    Reservoir* a_Reservoirs, 
    unsigned a_Width, 
    unsigned a_Height, 
    cudaSurfaceObject_t a_OutputBuffer);

__global__ void ShadeInternal(
    Reservoir* a_Reservoirs, 
    unsigned a_Width, 
    unsigned a_Height, 
    cudaSurfaceObject_t a_OutputBuffer);

//Shade all reservoirs for a specific pixel.
__device__ __forceinline__ void ShadeReservoirs(
    Reservoir* a_Reservoirs, 
    unsigned a_Width, 
    unsigned a_InputX, 
    unsigned a_InputY, 
    unsigned a_OutputX, 
    unsigned a_OutputY, 
    cudaSurfaceObject_t a_OutputBuffer);

/*
 * Generate the shadow rays used for shading.
 */
//__host__ unsigned int GenerateReSTIRShadowRaysShading(MemoryBuffer* a_AtomicBuffer, Reservoir* a_Reservoirs, const WaveFront::SurfaceData* a_PixelData, uint2 a_Resolution);

//__global__ void GenerateShadowRayShading(WaveFront::AtomicBuffer<RestirShadowRayShading>* a_AtomicBuffer, Reservoir* a_Reservoirs, const  WaveFront::SurfaceData* a_PixelData, uint2 a_Resolution, unsigned a_Depth);

/*
 * Resample spatial neighbours with an intermediate output buffer.
 * Combine reservoirs.
 *
 * Returns a pointer to the swap buffer containing the final combined reservoirs.
 */
__host__ Reservoir* SpatialNeighbourSampling(
    Reservoir* a_InputReservoirs,
    Reservoir* a_SwapBuffer1,
    Reservoir* a_SwapBuffer2,
    const WaveFront::SurfaceData* a_PixelData,
    const std::uint32_t a_Seed,
    uint2 a_Dimensions
);

__global__ void SpatialNeighbourSamplingInternal(
    Reservoir* a_ReservoirsIn,
    Reservoir* a_ReservoirsOut,
    const WaveFront::SurfaceData* a_PixelData,
    const std::uint32_t a_Seed,
    uint2 a_Dimensions,
    unsigned a_NumReservoirs
);

/*
 * Resample temporal neighbours and combine reservoirs.
 */
__host__ void TemporalNeighbourSampling(
    Reservoir* a_CurrentReservoirs,
    Reservoir* a_PreviousReservoirs,
    const WaveFront::SurfaceData* a_CurrentPixelData,
    const WaveFront::SurfaceData* a_PreviousPixelData,
    const std::uint32_t a_Seed,
    uint2 a_Dimensions,
    const cudaSurfaceObject_t a_MotionVectorBuffer,
    cudaSurfaceObject_t a_OutputBuffer
);

__global__ void CombineTemporalSamplesInternal(
    Reservoir* a_CurrentReservoirs,
    Reservoir* a_PreviousReservoirs,
    const WaveFront::SurfaceData* a_CurrentPixelData,
    const WaveFront::SurfaceData* a_PreviousPixelData,
    const std::uint32_t a_Seed,
    unsigned a_NumPixels,
    uint2 a_Dimensions,
    const cudaSurfaceObject_t a_MotionVectorBuffer,
    cudaSurfaceObject_t a_OutputBuffer
);

/*
 * Combine multiple reservoirs unbiased.
 * Stores the result in the reservoirs with the given pixel index.
 *
 * a_Count indicates the amount of pixels to process in a_ToCombine.
 */
__device__ __inline__ void CombineUnbiased(
    Reservoir* a_OutputReservoir, 
    const WaveFront::SurfaceData* a_OutputSurfaceData, 
    int a_Count, Reservoir* a_Reservoirs, 
    const WaveFront::SurfaceData* a_SurfaceData, 
    const std::uint32_t a_Seed);

/*
 * Combine multiple reservoirs biased.
 * Stores the result in the reservoirs with the given pixel index.
 *
 * a_Count indicates the amount of indices to process in a_ToCombineIndices.
 */
__device__ __inline__ void CombineBiased(
    Reservoir* a_OutputReservoir, 
    int a_Count, Reservoir* 
    a_Reservoirs, 
    const WaveFront::SurfaceData* a_SurfaceData, 
    const std::uint32_t a_Seed);

/*
 * Resample an old light sample.
 * This takes the given input, and resamples it against the given pixel data.
 * The output is then stored in a_Output.
 *
 */
__device__ __inline__ void Resample(
    LightSample* a_Input, 
    const WaveFront::SurfaceData* a_PixelData, 
    LightSample* a_Output);

///*
// * Generate shadow rays from the reservoirs after ReSTIR runs.
// * Add the shadow rays with scaled-by-weight payload to the wavefront buffer.
// */
//__host__ void GenerateWaveFrontShadowRays(Reservoir* a_Reservoirs, const WaveFront::SurfaceData* a_PixelData, WaveFront::AtomicBuffer<WaveFront::ShadowRayData>* a_AtomicBuffer, unsigned a_NumPixels);
//
//__global__ void GenerateWaveFrontShadowRaysInternal(Reservoir* a_Reservoirs, const WaveFront::SurfaceData* a_PixelData, WaveFront::AtomicBuffer<WaveFront::ShadowRayData>* a_AtomicBuffer, unsigned a_NumPixels, unsigned a_Depth);

/*
 * Combine reservoirs of two buffers.
 * Stores the combined results into the first reservoir buffer.
 */
__host__ void CombineReservoirBuffers(
    Reservoir* a_Reservoirs1, 
    Reservoir* a_Reservoirs2, 
    const WaveFront::SurfaceData* a_SurfaceData, 
    unsigned a_NumReservoirs, 
    unsigned a_Seed);

__global__ void CombineReservoirBuffersInternal(
    Reservoir* a_Reservoirs1, 
    Reservoir* a_Reservoirs2, 
    const WaveFront::SurfaceData* a_SurfaceData, 
    unsigned a_NumReservoirs, 
    unsigned a_Seed);

__device__ __inline__ uint32_t __mysmid();