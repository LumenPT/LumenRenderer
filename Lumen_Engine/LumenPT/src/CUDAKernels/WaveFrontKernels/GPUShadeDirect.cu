#include "GPUShadingKernels.cuh"
#include "../VolumetricKernels/GPUVolumetricShadingKernels.cuh"
#include "../../Shaders/CppCommon/RenderingUtility.h"
#include "../../Shaders/CppCommon/Half4.h"
#include <device_launch_parameters.h>
#include <sutil/vec_math.h>
#include "../disney.cuh"



CPU_ON_GPU void ResolveDirectLightHits(
    const SurfaceData* a_SurfaceDataBuffer,
    const uint2 a_Resolution,
    cudaSurfaceObject_t a_Output
)
{

    const unsigned int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    if(pixelX < a_Resolution.x && pixelY < a_Resolution.y)
    {
        const unsigned int pixelDataIndex = PIXEL_DATA_INDEX(pixelX, pixelY, a_Resolution.x);

        auto& pixelData = a_SurfaceDataBuffer[pixelDataIndex];
        //If the surface is emissive, store its light directly in the output buffer.
        if (pixelData.m_SurfaceFlags & SURFACE_FLAG_EMISSIVE)
        {

            half4Ushort4 color{pixelData.m_MaterialData.m_Color};
            surf2Dwrite<ushort4>(
                color.m_Ushort4,
                a_Output,
                pixelX * sizeof(ushort4),
                pixelY,
                cudaBoundaryModeTrap);
        }

    }
}

CPU_ON_GPU void ShadeDirect(
    const uint3 a_ResolutionAndDepth,
    const SurfaceData* a_SurfaceDataBuffer,
    const VolumetricData* a_VolumetricDataBuffer,
    const AtomicBuffer<TriangleLight>* const a_Lights,
    const unsigned a_Seed,
    const CDF* const a_CDF,
    AtomicBuffer<ShadowRayData>* const a_ShadowRays,
    AtomicBuffer<ShadowRayData>* const a_VolumetricShadowRays,
	cudaSurfaceObject_t a_VolumetricOutput
)
{

    const unsigned int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    const unsigned int pixelDataIndex = PIXEL_DATA_INDEX(pixelX, pixelY, a_ResolutionAndDepth.x);

    auto seed = WangHash(a_Seed + pixelDataIndex);

    if(pixelX < a_ResolutionAndDepth.x && pixelY < a_ResolutionAndDepth.y)
    {

        //TODO: return some form of light transform factor after resolving the distances in the volume.
        VolumetricShadeDirect(
            { pixelX, pixelY }, 
            a_ResolutionAndDepth, 
            a_VolumetricDataBuffer, 
            a_VolumetricShadowRays, 
            a_Lights,
            seed,
            a_CDF, 
            a_VolumetricOutput);

        // Get intersection.
        
        const SurfaceData& surfaceData = a_SurfaceDataBuffer[pixelDataIndex];

        if (!surfaceData.m_SurfaceFlags)
        {
            //Pick a light from the CDF.
            unsigned index;
            float pdf;
            a_CDF->Get(RandomFloat(seed), index, pdf);

            auto& light = *a_Lights->GetData(index);

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
                return;
            }

            //Geometry term G(x).
            const float solidAngle = (cosOut * light.area) / (lDistance * lDistance);

            
        	
            float bsdfPdf = 0.f;
            const auto bsdf = EvaluateBSDF(surfaceData.m_MaterialData, surfaceData.m_Normal, surfaceData.m_Tangent, -surfaceData.m_IncomingRayDirection, pixelToLightDir, bsdfPdf);
        	
            //If no contribution, don't make a shadow ray.
            if(bsdfPdf <= EPSILON)
            {
                return;
            }

            ////BSDF is equal to material color for now.
            //const auto brdf = MicrofacetBRDF(pixelToLightDir, -surfaceData.m_IncomingRayDirection, surfaceData.m_Normal,
            //                                 surfaceData.m_ShadingData.color, surfaceData.m_Metallic, surfaceData.m_Roughness);

            //The unshadowed contribution (contributed if no obstruction is between the light and surface) takes the BRDF,
            //geometry factor and solid angle into account. Also the light radiance.
            //The only thing missing from this is the scaling with the rest of the scene based on the reservoir PDF.
            float3 unshadowedPathContribution = (bsdf / bsdfPdf) * solidAngle * cosIn * light.radiance;

            //Scale by the PDF of this light to compensate for all other lights not being picked.
            unshadowedPathContribution *= ((1.f/pdf) * surfaceData.m_TransportFactor);

            /*
             * NOTE: Channel is Indirect because this runs at greater depth. This is direct light for an indirect bounce.
             */
            ShadowRayData shadowRay(
                PixelIndex{ pixelX, pixelY },
                surfaceData.m_Position,
                pixelToLightDir,
                lDistance - 0.2f,
                unshadowedPathContribution,
                LightChannel::INDIRECT);

            a_ShadowRays->Add(&shadowRay);
        }
    }
}