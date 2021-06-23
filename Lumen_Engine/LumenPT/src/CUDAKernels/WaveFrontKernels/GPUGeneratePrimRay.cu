#include "GPUShadingKernels.cuh"
#include "../../Shaders/CppCommon/SceneDataTableAccessor.h"
#include <device_launch_parameters.h>
#include "../../Shaders/CppCommon/Half4.h"



GPU_ONLY void HaltonSequence(
    unsigned int index,
    unsigned int base,
    float* result)
{
    ++index;

    float f = 1.f;
    float r = 0.f;

    while (index > 0)
    {
        f = f / base;
        r = r + f * (index % base);
        index = index / base;
    }

    *result = r;
}

CPU_ON_GPU void GeneratePrimaryRay(
    int a_NumRays,
    AtomicBuffer<IntersectionRayData>* const a_Buffer,
    float3 a_U,
    float3 a_V,
    float3 a_W,
    float3 a_Eye,
    uint2 a_Dimensions,
    unsigned int a_FrameCount,
    cudaSurfaceObject_t a_JitterOutput)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < a_NumRays; i += stride)
    {
        //Convert the index into the screen dimensions.
        const int screenY = i / a_Dimensions.x;
        const int screenX = i - (screenY * a_Dimensions.x);

        const unsigned int pixelDataIndex = PIXEL_DATA_INDEX(screenX, screenY, a_Dimensions.x);

    	float2 jitter;
    	HaltonSequence(a_FrameCount + static_cast<unsigned int>(i), 2, &jitter.x);
        HaltonSequence(a_FrameCount + static_cast<unsigned int>(i), 3, &jitter.y);

        //surf2Dwrite<float2>(
        //    jitter,
        //    a_JitterOutput,
        //    screenX * sizeof(float2),
        //    screenY,
        //    cudaBoundaryModeTrap
        //    );

        //half4Ushort4 jitter4 = {make_float4(jitter.x, jitter.y, 0.f, 0.f)};
        //surf2Dwrite<ushort4>(
        //    jitter4.m_Ushort4,
        //    a_JitterOutput,
        //    screenX * sizeof(ushort4),
        //    screenY,
        //    cudaBoundaryModeTrap
        //    );

        float3 direction = make_float3(static_cast<float>(screenX + jitter.x) / a_Dimensions.x,
                                       static_cast<float>(screenY + jitter.y) / a_Dimensions.y, 0.f);
        float3 origin = a_Eye;

        direction.x = -(direction.x * 2.0f - 1.0f);
        direction.y = -(direction.y * 2.0f - 1.0f);
        direction = normalize(direction.x * a_U + direction.y * a_V + a_W);

        IntersectionRayData ray{{screenX, screenY}, origin, direction, make_float3(1.f, 1.f, 1.f) };
        a_Buffer->Set(i, &ray); //Set because primary rays are ALWAYS a ray per pixel. No need to do atomic indexing. The atomic counter is manually set later.
    }
}
