#pragma once

/*
 * This file contains data structures used by ReSTIR on the GPU.
 */

#include <Optix/optix.h>
#include "../Shaders/CppCommon/CudaDefines.h"
#include <Cuda/cuda/helpers.h>
#include <assert.h>
#include <limits>

constexpr float MINFLOAT = std::numeric_limits<float>::min();


 /*
  * LightSample contains information about a light, specific to a surface in the scene.
  */
struct LightSample
{
    LightSample() : radiance({0.f, 0.f, 0.f}), normal({ 0.f, 0.f, 0.f }), position({ 0.f, 0.f, 0.f }), solidAnglePdf(0.f) {}

    float3 radiance;
    float3 normal;
    float3 position;
    float solidAnglePdf;
};

/*
 * An emissive triangle.
 */
struct TriangleLight
{
    float3 p0, p1, p2;
    float3 normal;
    float3 radiance;    //Radiance has the texture color baked into it. Probably average texture color.
};

/*
 * Reservoirs contain a weight and chosen sample.
 * A sample can be any object that has a weight.
 */
struct Reservoir
{
    GPU_ONLY Reservoir() : weightSum(0.f), sampleCount(0), weight(0.f)
    {

    }

    /*
     * Update this reservoir with a light and a weight for that light relative to the total set.
     */
    GPU_ONLY bool Update(const LightSample& a_Sample, float a_Weight)
    {
        assert(a_Weight >= 0.f);

        //Append weight to total.
        weightSum += a_Weight;
        ++sampleCount;

        //Generate a random float between 0 and 1 and then overwrite the sample based on the probability of this sample being chosen.
        const float r = static_cast<float>(rand()) / static_cast <float> (RAND_MAX);

        //In this case R is inclusive with 0.0 and 1.0. This means that the first sample is always chosen.
        //If weight is 0, then a division by 0 would happen. Also it'd be impossible to pick this sample.
        if (a_Weight != 0.f && r <= (a_Weight / weightSum))
        {
            sample = a_Sample;
            return true;
        }

        return false;
    }

    /*
     * Calculate the weight for this reservoir.
     */
    GPU_ONLY void UpdateWeight()
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

    GPU_ONLY void Reset()
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
    __device__ void Reset()
    {
        sum = 0.f;
        size = 0;
    }

    /*
     * Insert an element into the CDF.
     * The element will have the index of the current CDF size.
     * The weight is appended to the total weight sum and size is incremented by one.
     */
    __device__ void Insert(float a_Weight)
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
    __device__ void Get(float a_Value, unsigned& a_LightIndex, float& a_LightPdf)
    {
        //Index is not normalized in the actual set.
        int index = static_cast<int>(sum * a_Value);

        //Binary search
        int entry = BinarySearch(0, size - 1, index);

        float higher = data[entry];
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
     *
     */
    __device__ int BinarySearch(int a_First, int a_Last, float a_Value)
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
    unsigned lightIndex;
    float pdf;
};