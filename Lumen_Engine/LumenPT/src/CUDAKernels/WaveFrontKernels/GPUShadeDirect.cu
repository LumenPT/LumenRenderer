#include "GPUShadingKernels.cuh"
#include <device_launch_parameters.h>
#include <sutil/vec_math.h>

#include "../../Shaders/CppCommon/RenderingUtility.h"

CPU_ON_GPU void ShadeDirect(
    const uint3 a_ResolutionAndDepth,
    const SurfaceData* a_TemporalSurfaceDatBuffer,
    const SurfaceData* a_SurfaceDataBuffer,
    AtomicBuffer<ShadowRayData>* const a_ShadowRays,
    const TriangleLight* const a_Lights,
    const unsigned a_Seed,
    const unsigned a_CurrentDepth,
    const CDF* const a_CDF
)
{
    const unsigned int numPixels = a_ResolutionAndDepth.x * a_ResolutionAndDepth.y;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    auto seed = WangHash(a_Seed + index);

    //Handles cases where there are less threads than there are pixels.
    //i becomes index and is to be used by in functions where you need the pixel index.
    //i will update to a new pixel index if there is less threads than there are pixels.
    for (unsigned int i = index; i < numPixels; i += stride)
    {
        // Get intersection.
        const SurfaceData& surfaceData = a_SurfaceDataBuffer[i];

        if (surfaceData.m_IntersectionT > 0.f)
        {
            //Pick a light from the CDF.
            unsigned index;
            float pdf;
            a_CDF->Get(RandomFloat(seed), index, pdf);

            auto light = a_Lights[index];

            //Pick random point on light.
            const float u = RandomFloat(seed);
            const float v = RandomFloat(seed) * (1.f - u);
            float3 arm1 = light.p1 - light.p0;
            float3 arm2 = light.p2 - light.p0;
            float3 lightCenter = light.p0 + (arm1 * u) + (arm2 * v);
            float3 pixelPosition = surfaceData.m_Position;

            float3 pixelToLightDir = lightCenter - pixelPosition;
            //Direction from pixel to light.
            const float lDistance = length(pixelToLightDir);
            //Light distance from pixel.
            pixelToLightDir /= lDistance;
            //Normalize.
            const float cosIn = fmax(dot(pixelToLightDir, surfaceData.m_Normal), 0.f);

            const float cosOut = fmax(0.f, dot(light.normal, -pixelToLightDir));

            //Light normal at sample point dotted with light direction. Invert light dir for this (light to pixel instead of pixel to light)

            //Light is not facing towards the surface or too close to the surface.
            if (cosIn <= 0.f || lDistance <= 0.01f)
            {
                continue;
            }

            //Geometry term G(x).
            const float solidAngle = (cosOut * light.area) / (lDistance * lDistance);

            //BSDF is equal to material color for now.
            const auto brdf = MicrofacetBRDF(pixelToLightDir, -surfaceData.m_IncomingRayDirection, surfaceData.m_Normal,
                                             surfaceData.m_Color, surfaceData.m_Metallic, surfaceData.m_Roughness);

            //The unshadowed contribution (contributed if no obstruction is between the light and surface) takes the BRDF,
            //geometry factor and solid angle into account. Also the light radiance.
            //The only thing missing from this is the scaling with the rest of the scene based on the reservoir PDF.
            auto unshadowedPathContribution = brdf * solidAngle * cosIn * light.radiance;

            //Scale by the PDF of this light to compensate for all other lights not being picked.
            unshadowedPathContribution *= ((1.f/pdf) * surfaceData.m_TransportFactor);

            /*
             * NOTE: Channel is Indirect because this runs at greater depth. This is direct light for an indirect bounce.
             */
            ShadowRayData shadowRay(
                i,
                surfaceData.m_Position,
                pixelToLightDir,
                lDistance - 0.2f,
                unshadowedPathContribution,
                LightChannel::INDIRECT);

            a_ShadowRays->Add(&shadowRay);
        }
    }
}