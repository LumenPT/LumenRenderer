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
