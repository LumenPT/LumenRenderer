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
    const unsigned int a_NumLights,
    const CDF* const a_CDF )
{

    const unsigned int numPixels = a_ResolutionAndDepth.x * a_ResolutionAndDepth.y;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    //keep CDF pointer here
    //extract light index from CDF. just random float or smth

    //PDF probability density function

    //calculate potential radiance vector from PDF and BRDF

    //Handles cases where there are less threads than there are pixels.
    //i becomes index and is to be used by in functions where you need the pixel index.
    //i will update to a new pixel index if there is less threads than there are pixels.
    for (unsigned int i = index; i < numPixels; i += stride)
    {

        // Get intersection.
        const SurfaceData& surfaceData = a_SurfaceDataBuffer[i];

        if (surfaceData.m_IntersectionT > 0.f)
        {

            for(unsigned int lightIndex = 0; lightIndex < a_NumLights && a_NumLights <= 7; ++lightIndex)
            {
                const TriangleLight& light = a_Lights[lightIndex];

                float3 lightCenter = (light.p0 + light.p1 + light.p2) / 3.f;
                float3 pixelPosition = surfaceData.m_Position;

                //float3 shadowRayDir = surfaceData.m_Position - lightCenter;
                //const float lightDistance = length(shadowRayDir);
                //shadowRayDir = shadowRayDir / lightDistance;

                //float cosFactor = dot(-shadowRayDir, surfaceData.m_Normal);
                //cosFactor = fmax(cosFactor, 0.f);

                //float3 irradiance = cosFactor * (light.radiance / (lightDistance * lightDistance));

                ////HACK: apply epsilon... very hacky, very temporary
                //float3 shadowRayOrigin = surfaceData.m_Position - (shadowRayDir * 0.01f);

                float3 pixelToLightDir = lightCenter - pixelPosition;
                //Direction from pixel to light.
                const float lDistance = length(pixelToLightDir);
                //Light distance from pixel.
                pixelToLightDir /= lDistance;
                //Normalize.
                const float cosIn = fmax(dot(pixelToLightDir, surfaceData.m_Normal), 0.f);
                //Lambertian term clamped between 0 and 1. SurfaceN dot ToLight
                const float cosOut = 1.f; //This is a point light, which means that the normal is always pointing to the surface.
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
                const auto unshadowedPathContribution = brdf * solidAngle * cosIn * light.radiance;

                float3 shadowRayOrigin = surfaceData.m_Position + (surfaceData.m_Normal * 0.2f);

                ShadowRayData shadowRay(
                    i,
                    shadowRayOrigin,
                    pixelToLightDir,
                    lDistance - 0.2f,
                    unshadowedPathContribution,
                    LightChannel::DIRECT);

                a_ShadowRays->Add(&shadowRay);

            }

        }

    }
}