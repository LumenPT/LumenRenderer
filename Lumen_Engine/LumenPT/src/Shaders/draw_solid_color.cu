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


#include "../../vendor/Include/sutil/vec_math.h"
#include "Optix/optix.h"
#include "CppCommon/LaunchParameters.h"
#include "CppCommon/RenderingUtility.h"

extern "C" {
__constant__ LaunchParameters params;
}

extern "C"
__global__ void __raygen__draw_solid_color()
{
    uint3 launch_index = optixGetLaunchIndex();

    RaygenData* rgd = reinterpret_cast<RaygenData*>(optixGetSbtDataPointer());

    //float3 origin = make_float3(static_cast<float>(launch_index.x) / params.m_ImageWidth, static_cast<float>(launch_index.y) / params.m_ImageHeight, 0.0f);
    //origin.x = -(origin.x * 2.0f - 1.0f); //we inverse the result, because U image coordinate points left while X vector points right
    //origin.y = -(origin.y * 2.0f - 1.0f); //we inverse the result, because V image coordinate points down while Y vector points up
    //origin = origin.x * params.U + origin.y * params.V;
    //origin += params.eye;
    //float3 dir = params.W;
	
    float3 origin = make_float3(0.f);
    float3 dir = make_float3(0.f);

    orthgraphicProjection(
        origin,
        dir,
        make_int2(launch_index.x, launch_index.y),
        make_int2(params.m_ImageWidth, params.m_ImageHeight),
        params.eye,
        params.U,
        params.V,
        params.W
    );

    /*perspectiveProjection(
        origin,
        dir,
        make_int2(launch_index.x, launch_index.y),
        make_int2(params.m_ImageWidth, params.m_ImageHeight),
        params.eye,
        params.U,
        params.V,
        params.W
    );*/
	
    unsigned int p0, p1, p2;

    optixTrace(params.m_Handle, origin, dir, 0.0f, 1000.0f, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0, p0, p1, p2);

    float3 col = rgd->m_Color;

    float U = int_as_float(p1);
    float V = int_as_float(p2);

    Vertex* A = &params.m_VertexBuffer[p0 + 0];
    Vertex* B = &params.m_VertexBuffer[p0 + 1];
    Vertex* C = &params.m_VertexBuffer[p0 + 2];

    float W = 1.0f - (U + V);

    col = A->m_Normal * U + B->m_Normal * V + C->m_Normal * W;
    //col = make_float3(int_as_float(p0), int_as_float(p1), int_as_float(p2));

    if (p0 == 1)
    {
        col = make_float3(0.4f, 0.5f, 0.9f);
    }

    params.m_Image[launch_index.y * params.m_ImageWidth + launch_index.x] =
        make_color( col );
}

extern "C"
__global__ void __miss__MissShader()
{
    MissData* msd = reinterpret_cast<MissData*>(optixGetSbtDataPointer());


    optixSetPayload_0(1);
    optixSetPayload_1(float_as_int(msd->m_Color.y));
    optixSetPayload_2(float_as_int(msd->m_Color.z));
}

extern "C"
__global__ void __closesthit__HitShader()
{
    const float2 barycentrics = optixGetTriangleBarycentrics();
    HitData* msd = reinterpret_cast<HitData*>(optixGetSbtDataPointer());;
    auto col = make_float4(0.0f, 1.0f, 0.0f, 1.0f);
    auto col1 = tex2D<float4>(msd->m_TextureObject, barycentrics.x, 1 - barycentrics.y);


    optixSetPayload_0(optixGetPrimitiveIndex());
    optixSetPayload_1(float_as_int(barycentrics.x));
    optixSetPayload_2(float_as_int(barycentrics.y));
}
