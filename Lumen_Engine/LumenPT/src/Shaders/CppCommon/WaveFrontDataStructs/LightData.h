#pragma once
#include "../CudaDefines.h"

#include <Optix/optix.h>
#include <Cuda/cuda/helpers.h>
#include <cassert>

namespace WaveFront
{

    enum class LightChannel : unsigned int
    {
        DIRECT,
        INDIRECT,
        SPECULAR,
        NUM_CHANNELS
    };

    //An emissive triangle
    struct TriangleLight
    {
        float3 p0, p1, p2;
        float3 normal;
        float3 radiance;    //Radiance has the texture color baked into it. Probably average texture color.
        float area;         //The area of the triangle. This is required to project the light onto a hemisphere.
    };

}
