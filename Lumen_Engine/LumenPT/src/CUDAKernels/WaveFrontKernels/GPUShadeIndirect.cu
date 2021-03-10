#include "GPUShadingKernels.cuh"
#include <device_launch_parameters.h>

CPU_ON_GPU void ShadeIndirect(
    const uint3 a_ResolutionAndDepth,
    const IntersectionRayBatch* const a_PrimaryRays,  //TODO: These need to be the rays of that the current wave intersected before calling this wave.
    const IntersectionBuffer* const a_Intersections,
    IntersectionRayBatch* const a_Output)
{

    const unsigned int numPixels = a_ResolutionAndDepth.x * a_ResolutionAndDepth.y;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (int i = index; i < numPixels; i += stride)
    {

        //Convert the index into the screen dimensions.
        const int screenY = i / a_ResolutionAndDepth.x;
        const int screenX = i - (screenY * a_ResolutionAndDepth.x);

        const IntersectionData& intersection = a_Intersections->GetIntersection(i, 0);
        ;       const IntersectionRayData& ray = a_PrimaryRays->GetRay(i, 0);

        //TODO russian roulette to terminate path (multiply weight with russian roulette outcome
        float russianRouletteWeight = 1.f;

        //TODO extract surface normal from intersection data.
        float3 normal = make_float3(1.f, 0.f, 0.f);

        //TODO generate random direction
        float3 dir = make_float3(1.f, 0.f, 0.f);

        //TODO get position from intersection data.
        float3 pos = make_float3(0.f, 0.f, 0.f);

        //TODO calculate BRDF to see how much light is transported.
        float3 brdf = make_float3(1.f, 1.f, 1.f); //Do this after direct shading because the material is already looked up there. Use the BRDF.
        float3 totalLightTransport = russianRouletteWeight * ray.m_Contribution * brdf;  //Total amount of light that will end up in the camera through this path.

        //Add to the output buffer.
        a_Output->SetRay({ pos, dir, totalLightTransport }, i, 0);
    }

}