#include "GPUShadingKernels.cuh"
#include <device_launch_parameters.h>

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
    IntersectionRayData* const a_Buffer,
    float3 a_U,
    float3 a_V,
    float3 a_W,
    float3 a_Eye,
    int2 a_Dimensions,
    unsigned int a_FrameCount)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < a_NumRays; i += stride)
    {

        //Convert the index into the screen dimensions.
        const int screenY = i / a_Dimensions.x;
        const int screenX = i - (screenY * a_Dimensions.x);

    	float2 jitter;
    	HaltonSequence(a_FrameCount + static_cast<unsigned int>(i), 2, &jitter.x);
    	HaltonSequence(a_FrameCount + static_cast<unsigned int>(i), 3, &jitter.y);
    	
        float3 direction = make_float3(static_cast<float>(screenX + jitter.x) / a_Dimensions.x,
                                       static_cast<float>(screenY + jitter.y) / a_Dimensions.y, 0.f);
        float3 origin = a_Eye;

        direction.x = -(direction.x * 2.0f - 1.0f);
        direction.y = -(direction.y * 2.0f - 1.0f);
        direction = normalize(direction.x * a_U + direction.y * a_V + a_W);

        IntersectionRayData ray{ origin, direction, make_float3(1.f, 1.f, 1.f) };
        a_Buffer[i] = ray;
    }
}

CPU_ON_GPU void ExtractSurfaceDataGpu(unsigned a_NumIntersections, AtomicBuffer<IntersectionData>* a_IntersectionData, AtomicBuffer<IntersectionRayData>* a_Rays, SurfaceData* a_OutPut, DeviceMaterial* a_Materials)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < a_NumIntersections; i += stride)
    {
        //TODO: ensure that index is the same for the intersection data and ray.
        auto* intersection = a_IntersectionData->GetData(i);
        auto* ray = a_Rays->GetData(i);

        //TODO get material pointer from intersection data.
        //TODO extract barycentric coordinates and all that.
        //TODO store in output buffer.
    }

}
