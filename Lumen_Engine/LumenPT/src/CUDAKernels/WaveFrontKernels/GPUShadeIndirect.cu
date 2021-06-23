#include "GPUShadingKernels.cuh"
#include <device_launch_parameters.h>

#include "../../Shaders/CppCommon/RenderingUtility.h"
#include "../disney.cuh"

CPU_ON_GPU void ShadeIndirect(
    const uint3 a_ResolutionAndDepth,
    const SurfaceData* a_SurfaceDataBuffer,
    AtomicBuffer<IntersectionRayData>* a_IntersectionRays,
    const unsigned a_Seed
)
{
    const unsigned int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    const unsigned int pixelDataIndex = PIXEL_DATA_INDEX(pixelX, pixelY, a_ResolutionAndDepth.x);

    //Outside of loop because multiple items can be processed by one thread. RandomFloat modifies the seed from within the loop so no repetition occurs.
    auto seed = WangHash(a_Seed + WangHash(pixelDataIndex));

    //Loop over the amount of intersections.
    if (pixelX < a_ResolutionAndDepth.x && pixelY < a_ResolutionAndDepth.y)
    {
        
        auto& surfaceData = a_SurfaceDataBuffer[pixelDataIndex];

    	//Check for alpha discarded surfaces. If alpha discarded, just continue in the same direction.
    	if(surfaceData.m_SurfaceFlags & SURFACE_FLAG_ALPHA_TRANSPARENT)
    	{
            IntersectionRayData ray{ {pixelX, pixelY}, surfaceData.m_Position, surfaceData.m_IncomingRayDirection, surfaceData.m_TransportFactor};
            a_IntersectionRays->Add(&ray);
            return;
    	}

        //If the surface is emissive or not intersected, terminate.
        if(surfaceData.m_SurfaceFlags)
        {
            return;
        }

        /*
         * If the angle is too perpendicular to the normal, discard. It's too prone to floating point error which means it can't generate a reflection
         * some of the time.
         */
        if (fabs(dot(surfaceData.m_Normal, surfaceData.m_IncomingRayDirection)) < 3.f * EPSILON)
        {
            return;
        }

        ////TODO replace with BSDF sampling (needs tangent).
        ////Calculate a diffuse reflection direction based on the surface roughness. Also retrieves the PDF for that direction being chosen on the full sphere.
        //float brdfPdf;
        //float3 bounceDirection;
        //SampleHemisphere(surfaceData.m_IncomingRayDirection, surfaceData.m_Normal, ROUGHNESS, seed, bounceDirection, brdfPdf);

        //if(brdfPdf <= 0)
        //{
        //    printf("Bad PDF: %f\n", brdfPdf);
        //}

        //assert(!isnan(bounceDirection.x) && !isnan(bounceDirection.y) && !isnan(bounceDirection.z));
        //assert(!isnan(brdfPdf));
        //assert(brdfPdf >= 0.f);

        /*
         * Terminate on inter-reflections. This is common for diffuse surfaces but not so much for others.
         * The angle of incidence makes a big difference too.
         * This could be resolved with RIS, by doing four strata on the hemisphere and taking four samples.
         * At least three of the samples will be correct.
         * This does however require multiple BRDF evaluations which is expensive.
         *
         * Because half the domain is removed, the BRDF PDF can be doubled after this passes.
         */
        /*const auto bounceDotN = dot(bounceDirection, surfaceData.m_Normal);
        if (bounceDotN <= 0.f) return;*/

        ////Double BRDF PDF because half the domain is terminated above.
        //brdfPdf *= 2.f;

        /*
         * Scale the path contribution based on the PDF (over 4 PI, the entire sphere).
         * When perfectly diffuse, 1/4pi will result in exactly scaling by 4pi.
         * When mirroring, a high PDF way larger than 1 will scale down the contribution because now it comes from just one direction.
         */
        //const auto brdf = MicrofacetBRDF(invViewDir, bounceDirection, surfaceData.m_Normal, shadingData.color, METALLIC, ROUGHNESS);
        //pathContribution *= ((brdf * bounceDotN) / brdfPdf);

        float3 bounceDirection;
        float pdf = 0.f;
        bool specular = false;
        const auto bsdf = SampleBSDF(
            surfaceData.m_MaterialData, 
            surfaceData.m_Normal, 
            surfaceData.m_Normal, 
            surfaceData.m_Tangent, 
            -surfaceData.m_IncomingRayDirection, 
            1.f, 
            RandomFloat(seed), 
            RandomFloat(seed), 
            RandomFloat(seed), 
            bounceDirection, 
            pdf, 
            specular);
    	
        //Skip rays that have a tiny PDF.
        if (pdf <= EPSILON || isnan(pdf + bsdf.x + bsdf.y + bsdf.z))
        {
            return;
        }

        

        //Apply russian roulette based on the BSDF. Mirrors always survive. For other surfaces take the max light transport channel and clamp at 1.
        const float russianRouletteWeight = specular ? 1.f : fminf(fmaxf(bsdf.x, fmaxf(bsdf.y, bsdf.z)), 1.f);
        const float rand = RandomFloat(seed);

        //Path termination.
        if (russianRouletteWeight < rand)
        {
            return;
        }

        
    	
        //Scale contribution up because the path survived.
        const float russianPdf = 1.f / russianRouletteWeight;
        assert(russianPdf >= 0.f);
        assert(surfaceData.m_TransportFactor.x >= 0 && surfaceData.m_TransportFactor.y >= 0 && surfaceData.m_TransportFactor.z >= 0);
        float3 pathContribution = surfaceData.m_TransportFactor * russianPdf;
    	
        //Add the BSDF to the path contribution, along with the angle of incidence scaling.
        //Also scale by the BSDF PDF right away.
        pathContribution *= bsdf * fabsf(dot(surfaceData.m_Normal, bounceDirection)) * (1.f/pdf);
    	
        assert(pathContribution.x >= 0 && pathContribution.y >= 0 && pathContribution.z >= 0);

        //Finally add the ray to the ray buffer.
        IntersectionRayData ray{{pixelX, pixelY}, surfaceData.m_Position, bounceDirection, pathContribution};

        a_IntersectionRays->Add(&ray);

        return; //Breaky
    }

}