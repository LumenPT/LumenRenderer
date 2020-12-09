//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//


#include <cuda/helpers.h>

#include "Optix/optix.h"
#include "CppCommon/LaunchParameters.h"

extern "C" {
__constant__ LaunchParameters params;
}

extern "C"
__global__ void __raygen__draw_solid_color()
{
    uint3 launch_index = optixGetLaunchIndex();

    RaygenData* rgd = reinterpret_cast<RaygenData*>(optixGetSbtDataPointer());

    float3 origin = make_float3(static_cast<float>(launch_index.x) / params.m_ImageWidth, static_cast<float>(launch_index.y) / params.m_ImageHeight, 0.0f);
    origin.x = origin.x * 2.0f - 1.0f;
    origin.y = origin.y * 2.0f - 1.0f;
    float3 dir = make_float3(0.0f, 0.0f, 1.0f);

    unsigned int p0;

    optixTrace(params.m_Handle, origin, dir, 0.0f, 1000.0f, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0, p0);

    float3 col = rgd->m_Color;

    if (p0 == 1)
    {
        col = make_float3(1.0f, 1.0f, 0.0f);
    }
    else if (p0 == 3)
    {
        col = make_float3(1.0f, 0.0f, 1.0f);
    }

    params.m_Image[launch_index.y * params.m_ImageWidth + launch_index.x] =
        make_color( col );
}

extern "C"
__global__ void __miss__MissShader()
{
    MissData* msd = reinterpret_cast<MissData*>(optixGetSbtDataPointer());;
    optixSetPayload_0(msd->m_Num);
}

extern "C"
__global__ void __closesthit__HitShader()
{
    optixSetPayload_0(1);
}
