#include "GPUShadingKernels.cuh"
#include <device_launch_parameters.h>
#include <sutil/vec_math.h>

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

                float3 shadowRayDir = surfaceData.m_Position - lightCenter;
                const float lightDistance = length(shadowRayDir);
                shadowRayDir = shadowRayDir / lightDistance;

                float cosFactor = dot(-shadowRayDir, surfaceData.m_Normal);
                cosFactor = fmax(cosFactor, 0.f);

                float3 irradiance = cosFactor * (light.radiance / (lightDistance * lightDistance));

                //HACK: apply epsilon... very hacky, very temporary
                float3 shadowRayOrigin = surfaceData.m_Position + (shadowRayDir * 0.01f);

                ShadowRayData shadowRay(
                    i,
                    shadowRayOrigin,
                    shadowRayDir,
                    lightDistance - 0.01f,
                    surfaceData.m_Color * irradiance * surfaceData.m_TransportFactor,
                    LightChannel::DIRECT);

                a_ShadowRays->Add(&shadowRay);

            }

        }

    }
}