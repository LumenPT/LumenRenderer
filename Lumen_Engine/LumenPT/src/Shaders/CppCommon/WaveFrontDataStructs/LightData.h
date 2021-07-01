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
        VOLUMETRIC,
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


    //Union used to store the data in a texture and read/write with easy conversions.
    union TriangleLightUint4_4
    {

        TriangleLight m_TriangleLight;
        uint4 m_Uint4[4];   //16 floats equal 16 uints in byte size. Requires 4 reads to be done.

    };

    const unsigned s = sizeof(TriangleLightUint4_4::m_Uint4);

}
