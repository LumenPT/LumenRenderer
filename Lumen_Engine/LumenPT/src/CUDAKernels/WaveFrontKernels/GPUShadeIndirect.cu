#include "GPUShadingKernels.cuh"
#include <device_launch_parameters.h>

#include "../../Shaders/CppCommon/RenderingUtility.h"

CPU_ON_GPU void ShadeIndirect(
    const uint3 a_ResolutionAndDepth,
    const float3 a_CameraPosition,
    const SurfaceData* a_SurfaceDataBuffer,
    const AtomicBuffer<IntersectionData>* a_Intersections,
    AtomicBuffer<IntersectionRayData>* a_IntersectionRays,
    const unsigned a_NumIntersections,
    const unsigned a_CurrentDepth,
    const unsigned a_Seed
)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    //Outside of loop because multiple items can be processed by one thread. RandomFloat modifies the seed from within the loop so no repetition occurs.
    auto seed = WangHash(a_Seed + WangHash(index));

    //Loop over the amount of intersections.
    for (int i = index; i < a_NumIntersections; i += stride)
    {
        auto& intersection = *a_Intersections->GetData(i);
        auto& surfaceData = a_SurfaceDataBuffer[intersection.m_PixelIndex];

        //If the surface is emissive or not intersected, terminate.
        if(surfaceData.m_Emissive || surfaceData.m_IntersectionT <= 0.f)
        {
            continue;
        }

        //Apply russian roulette based on the surface color (dark absorbs more, so terminates sooner).
        const float russianRouletteWeight = clamp(fmaxf(surfaceData.m_Color.x, fmaxf(surfaceData.m_Color.y, surfaceData.m_Color.z)), 0.f, 1.f);
        const float rand = RandomFloat(seed);
        
        //Path termination.
        if (russianRouletteWeight < rand)
        {
            continue;
        }

        assert(surfaceData.m_TransportFactor.x >= 0 && surfaceData.m_TransportFactor.y >= 0 && surfaceData.m_TransportFactor.z >= 0);

        //Scale contribution up because the path survived.
        const float russianPdf = 1.f / russianRouletteWeight;
        float3 pathContribution = surfaceData.m_TransportFactor * russianPdf;

        /*
         * If the angle is too perpendicular to the normal, discard. It's too prone to floating point error which means it can't generate a reflection
         * some of the time.
         */
        if(dot(surfaceData.m_Normal, surfaceData.m_IncomingRayDirection) >= -10.f * EPSILON)
        {
            continue;
        }

        //Calculate a diffuse reflection direction based on the surface roughness. Also retrieves the PDF for that direction being chosen on the full sphere.
        float brdfPdf;
        float3 bounceDirection;
        SampleHemisphere(surfaceData.m_IncomingRayDirection, surfaceData.m_Normal, surfaceData.m_Roughness, seed, bounceDirection, brdfPdf);

        if(brdfPdf <= 0)
        {
            printf("Bad PDF: %f\n", brdfPdf);
        }

        assert(!isnan(bounceDirection.x) && !isnan(bounceDirection.y) && !isnan(bounceDirection.z));
        assert(!isnan(brdfPdf));
        assert(brdfPdf >= 0.f);
        assert(russianPdf >= 0.f);

        /*
         * Terminate on inter-reflections. This is common for diffuse surfaces but not so much for others.
         * The angle of incidence makes a big difference too.
         * This could be resolved with RIS, by doing four strata on the hemisphere and taking four samples.
         * At least three of the samples will be correct.
         * This does however require multiple BRDF evaluations which is expensive.
         *
         * Because half the domain is removed, the BRDF PDF can be doubled after this passes.
         */
        const auto bounceDotN = dot(bounceDirection, surfaceData.m_Normal);
        if (bounceDotN <= 0.f) continue;

        //Double BRDF PDF because half the domain is terminated above.
        brdfPdf *= 2.f;

        /*
         * Scale the path contribution based on the PDF (over 4 PI, the entire sphere).
         * When perfectly diffuse, 1/4pi will result in exactly scaling by 4pi.
         * When mirroring, a high PDF way larger than 1 will scale down the contribution because now it comes from just one direction.
         */
        const auto invViewDir = -surfaceData.m_IncomingRayDirection;
        const auto brdf = MicrofacetBRDF(invViewDir, bounceDirection, surfaceData.m_Normal, surfaceData.m_Color, surfaceData.m_Metallic, surfaceData.m_Roughness);
        pathContribution *= ((brdf * bounceDotN) / brdfPdf);

        assert(pathContribution.x >= 0 && pathContribution.y >= 0 && pathContribution.z >= 0);

        //Finally add the ray to the ray buffer.
        IntersectionRayData ray{intersection.m_PixelIndex, surfaceData.m_Position, bounceDirection, pathContribution};
        a_IntersectionRays->Add(&ray);
    }

}