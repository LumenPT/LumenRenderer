#pragma once

/*
 * This file contains data structures used by ReSTIR on the GPU.
 */

#include "CudaDefines.h"
#include "WaveFrontDataStructs/LightDataBuffer.h"

#include <Optix/optix.h>
#include <Cuda/cuda/helpers.h>
#include <assert.h>
#include <limits>
#include <cinttypes>

#include "../../CUDAKernels/RandomUtilities.cuh"

constexpr float MINFLOAT = std::numeric_limits<float>::epsilon();

/*
 * All configurable settings for ReSTIR.
 * The defaults are the values used in the ReSTIR paper.
 */
struct ReSTIRSettings
{
    //Screen width in pixels.
    std::uint32_t width = 0;

    //Screen height in pixels.
    std::uint32_t height = 0;

    //The amount of reservoirs used per pixel.
    static constexpr std::uint32_t numReservoirsPerPixel = 1; //Default 5

    //The amount of lights per light bag.
    static constexpr std::uint32_t numLightsPerBag = 1000;  //Default 1000

    //The total amount of light bags to generate.
    static constexpr std::uint32_t numLightBags = 50;   //Default 50

    //The amount of initial samples to take each frame.
    static constexpr std::uint32_t numPrimarySamples = 32;  //Default 32

    //The amount of spatial neighbours to consider.
    static constexpr std::uint32_t numSpatialSamples = 5;   //Default 5    

    //The maximum distance for spatial samples.
    static constexpr std::uint32_t spatialSampleRadius = 30;    //Default 30

    //The x and y size of the pixel grid per light bag.
    static constexpr std::uint32_t pixelGridSize = 16;      //Default 16

    //The amount of spatial iterations to perform. Previous output becomes current input.
    //This has to be an even number so that the final results end up in the right buffer.
    static constexpr std::uint32_t numSpatialIterations = 2;    //Default 2

    //Use the biased algorithm or not. When false, the unbiased algorithm is used instead.
    static constexpr bool enableBiased = true;  //Default true

    //Enable spatial sampling.
    static constexpr bool enableSpatial = true; //Default true

    //Enable temporal sampling.
    static constexpr bool enableTemporal = true;    //Default true

    //Enable or disable certain parts of ReSTIR TODO
#define RESTIR_BIASED 1
#define RESTIR_SPATIAL 1
#define RESTIR_TEMPORAL 1
};

/*
 * A visibility ray used by ReSTIR to determine if a light sample is occluded or not.
 */
struct RestirShadowRay
{
    float3 origin;      //Ray origin.
    float3 direction;   //Ray direction, normalized.
    float distance;     //The distance of the light source from the origin along the ray direction.
    unsigned index;     //The index into the reservoir buffer at which to set weight to 0 when occluded.
};

 /*
  * LightSample contains information about a light, specific to a surface in the scene.
  */
struct LightSample
{
    __host__ __device__ LightSample() : radiance({0.f, 0.f, 0.f}), normal({ 0.f, 0.f, 0.f }), position({ 0.f, 0.f, 0.f }), area(0.f), solidAnglePdf(0.f) {}

    float3 radiance;
    float3 normal;
    float3 position;
    float area;
    float3 unshadowedPathContribution;       //Contribution with geometry and BSDF taken into account.
    float solidAnglePdf;                    //solid angle PDF which is used to weight this samples importance.
};

/*
 * Reservoirs contain a weight and chosen sample.
 * A sample can be any object that has a weight.
 */
struct Reservoir
{
    GPU_ONLY INLINE Reservoir() : weightSum(0.f), sampleCount(0), weight(0.f)
    {

    }

    /*
     * Update this reservoir with a light and a weight for that light relative to the total set.
     */
    GPU_ONLY INLINE bool Update(const LightSample& a_Sample, float a_Weight, std::uint32_t a_Seed)
    {
        assert(a_Weight >= 0.f);

        //Append weight to total.
        weightSum += a_Weight;
        ++sampleCount;

        //Generate a random float between 0 and 1 and then overwrite the sample based on the probability of this sample being chosen.
        const float r = RandomFloat(a_Seed);

        //In this case R is inclusive with 0.0 and 1.0. This means that the first sample is always chosen.
        //If weight is 0, then a division by 0 would happen. Also it'd be impossible to pick this sample.
        if (r <= (a_Weight / weightSum))
        {
            sample = a_Sample;
            return true;
        }

        return false;
    }

    /*
     * Calculate the weight for this reservoir.
     */
    GPU_ONLY INLINE void UpdateWeight()
    {
        //If no samples have been considered yet, then the weight is also 0 (prevents division by 0).
        //Also 0 if none of the considered samples contributed.
        if (sampleCount == 0 || weightSum <= 0.f)
        {
            weight = 0;
            return;
        }

        //Can't divide by 0 so take a super tiny PDF instead.
        weight = (1.f / fmaxf(sample.solidAnglePdf, MINFLOAT)) * ((1.f / static_cast<float>(sampleCount)) * weightSum);
    }

    GPU_ONLY INLINE void Reset()
    {
        weightSum = 0.f;
        sampleCount = 0;
        weight = 0.f;
    }

    //The sum of individual samples weight.
    float weightSum;

    //The amount of samples seen.
    long long sampleCount;

    //The actual resulting weight. Calculate using UpdateWeight function.
    float weight;
    LightSample sample;
};

struct CDF
{
    GPU_ONLY void Reset()
    {
        sum = 0.f;
        size = 0;
    }

    /*
     * Insert an element into the CDF.
     * The element will have the index of the current CDF size.
     * The weight is appended to the total weight sum and size is incremented by one.
     */
    GPU_ONLY void Insert(float a_Weight)
    {
        //Important: This is not thread safe. Build from a single thread.
        sum += a_Weight;
        data[size] = sum;
        ++size;
    }

    /*
     * Get the index and value stored for an input value.
     * The input value has to be normalized between 0.0 and 1.0, both inclusive.
     * The found element's index and PDF will be stored in the passed references.
     */
    GPU_ONLY void Get(float a_Value, unsigned& a_LightIndex, float& a_LightPdf) const
    {
        //Index is not normalized in the actual set.
        const float requiredValue = sum * a_Value;

        //Binary search
        const int entry = BinarySearch(0, size - 1, requiredValue);

        const float higher = data[entry];

        //TODO this if statement can be avoided by always making index  0 equal to 0. Then offset array indices by 1 and add 1 to the size req of the class.
        float lower = 0.f;
        if (entry != 0)
        {
            lower = data[entry - 1];
        }

        //Pdf is proportional to all entries in the dataset.
        a_LightIndex = entry;
        a_LightPdf = (higher - lower) / sum;
    }

    /*
     * Find the entry with the given value by doing a binary search.
     * This is a recursive function.
     */
    GPU_ONLY int BinarySearch(int a_First, int a_Last, float a_Value) const
    {
        assert(a_Value >= 0.f && a_Value <= sum && "Binary search key must be within set bounds.");
        assert(a_First >= 0 && a_First <= a_Last);

        //Get the middle element.
        const int center = a_First + (a_Last - a_First) / 2;

        //Upper and lower bound.
        float higher = data[center];
        float lower = 0.f;

        if(center != 0)
        {
            lower = data[center - 1];
        }

        //Element is smaller, so search in the lower half of the data range.
        if (a_Value < lower)
        {
            return BinarySearch(a_First, center - 1, a_Value);
        }

        //Bigger, so search in the upper half of the data range.
        if (a_Value > higher)
        {
            return BinarySearch(center + 1, a_Last, a_Value);
        }

        //The value lies between the lower and higher bound, so the current element is the right one.
        return center;
    }

    float sum;
    unsigned size;
    float data[];
};

struct LightBagEntry
{
    WaveFront::TriangleLight light;
    float pdf;
};