#include "GPUShadingKernels.cuh"
#include <device_launch_parameters.h>

CPU_ON_GPU void ShadeDirect(
    const uint3 a_ResolutionAndDepth,
    const IntersectionRayBatch* const a_CurrentRays,
    const IntersectionBuffer* const a_CurrentIntersections,
    ShadowRayBatch* const a_ShadowRays,
    const LightDataBuffer* const a_Lights,
    CDF* const a_CDF)
{

    const unsigned int numPixels = a_ResolutionAndDepth.x * a_ResolutionAndDepth.y;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    float3 potRadiance = { 0.5f,0.5f,0.5f };

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
        const IntersectionData& currIntersection = a_CurrentIntersections->GetIntersection(i, 0 /*There is only one ray per pixel, only one intersection per pixel (for now)*/);

        if (currIntersection.IsIntersection())
        {

            // Get ray used to calculate intersection.
            const unsigned int rayArrayIndex = currIntersection.m_RayArrayIndex;

            const IntersectionRayData& currRay = a_CurrentRays->GetRay(rayArrayIndex);

            //unsigned int lightIndex = 0;
            //float lightPDF = 0;
            //float random = RandomFloat(lightIndex);

            //auto& diffColor = currIntersection.m_Primitive->m_Material->m_DiffuseColor;

            //potRadiance = make_float3(diffColor);

            potRadiance = make_float3(1.f);

            float3 sRayOrigin = currRay.m_Origin + (currRay.m_Direction * currIntersection.m_IntersectionT);

            //const auto& light = a_Lights[lightIndex];
            //
            ////worldspace position of emissive triangle
            //const float3 lightPos = (light.m_Lights->p0 + light.m_Lights->p1 + light.m_Lights->p2) * 0.33f;
            //float3 sRayDir = normalize(lightPos - sRayOrigin);


            unsigned int x = threadIdx.x, y = threadIdx.y, z = threadIdx.z;
            float3 sRayDir = { RandomFloat(x), RandomFloat(y), RandomFloat(z) };
            sRayDir = normalize(sRayDir);

            // apply epsilon... very hacky, very temporary
            sRayOrigin = sRayOrigin + (sRayDir * 0.001f);

            ShadowRayData shadowRay(
                sRayOrigin,
                sRayDir,
                0.11f,
                potRadiance,    // light contribution value at intersection point that will reflect in the direction of the incoming ray (intersection ray).
                ResultBuffer::OutputChannel::DIRECT
            );

            a_ShadowRays->SetShadowRay(shadowRay, a_ResolutionAndDepth.z, i, 0 /*Ray index is 0 as there is only one ray per pixel*/);

        }

    }
}