#include "GPUShadingKernels.cuh"
#include "../../Shaders/CppCommon/SceneDataTableAccessor.h"
#include <Lumen/ModelLoading/MeshInstance.h>
#include <device_launch_parameters.h>
#include <sutil/vec_math.h>
#include "../disney.cuh"

CPU_ON_GPU void ExtractSurfaceDataGpu(
    unsigned a_NumIntersections,
    AtomicBuffer<IntersectionData>* a_IntersectionData,
    AtomicBuffer<IntersectionRayData>* a_Rays,
    SurfaceData* a_OutPut,
    uint2 a_Resolution,
    SceneDataTableAccessor* a_SceneDataTable)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < a_NumIntersections; i += stride)
    {
        const IntersectionData currIntersection = *a_IntersectionData->GetData(i);
        const IntersectionRayData currRay = *a_Rays->GetData(currIntersection.m_RayArrayIndex);
        unsigned int surfaceDataIndex = PIXEL_DATA_INDEX(currIntersection.m_PixelIndex.m_X, currIntersection.m_PixelIndex.m_Y, a_Resolution.x);

        if (currIntersection.IsIntersection())
        {
            // Get ray used to calculate intersection.
            const unsigned int instanceId = currIntersection.m_InstanceId;
            const unsigned int rayArrayIndex = currIntersection.m_RayArrayIndex;
            const unsigned int vertexIndex = 3 * currIntersection.m_PrimitiveIndex;

            assert(a_SceneDataTable->GetTableEntry<DevicePrimitive>(instanceId) != nullptr);
            const auto devicePrimitiveInstance = *a_SceneDataTable->GetTableEntry<DevicePrimitiveInstance>(instanceId);
            const auto devicePrimitive = devicePrimitiveInstance.m_Primitive;

            assert(devicePrimitive.m_Material != nullptr);
            assert(devicePrimitive.m_IndexBuffer != nullptr);
            assert(devicePrimitive.m_VertexBuffer != nullptr);

            const unsigned int vertexIndexA = devicePrimitive.m_IndexBuffer[vertexIndex + 0];
            const unsigned int vertexIndexB = devicePrimitive.m_IndexBuffer[vertexIndex + 1];
            const unsigned int vertexIndexC = devicePrimitive.m_IndexBuffer[vertexIndex + 2];

            const Vertex* A = &devicePrimitive.m_VertexBuffer[vertexIndexA];
            const Vertex* B = &devicePrimitive.m_VertexBuffer[vertexIndexB];
            const Vertex* C = &devicePrimitive.m_VertexBuffer[vertexIndexC];

            const float U = currIntersection.m_Barycentrics.x;
            const float V = currIntersection.m_Barycentrics.y;
            const float W = 1.f - (U + V);

            float2 texCoords = A->m_UVCoord * W + B->m_UVCoord * U + C->m_UVCoord * V;

            //Whether or not tangent flipping should happen.
            const auto flip = A->m_Tangent.w;

            const DeviceMaterial* material = devicePrimitive.m_Material;

            //TODO extract different textures (emissive, diffuse, metallic, roughness).
            const float4 normalMap = tex2D<float4>(material->m_NormalTexture, texCoords.x, texCoords.y);
            const float4 textureColor = tex2D<float4>(material->m_DiffuseTexture, texCoords.x, texCoords.y);
            const float3 finalColor = make_float3(textureColor * material->m_DiffuseColor);
            const float4 metalRoughness = tex2D<float4>(material->m_MetalRoughnessTexture, texCoords.x, texCoords.y);

            //Emissive mode
            float4 emissive = make_float4(0.f);

            //When enabled, just read the GLTF data and scale it accordingly.
            if (devicePrimitiveInstance.m_EmissionMode == Lumen::EmissionMode::ENABLED)
            {
                emissive = tex2D<float4>(material->m_EmissiveTexture, texCoords.x, texCoords.y);
                emissive *= material->m_EmissionColor * devicePrimitiveInstance.m_EmissiveColorAndScale.w;
            }

            //When override, take the ovverride emissive color and scale it up.
            else if (devicePrimitiveInstance.m_EmissionMode == Lumen::EmissionMode::OVERRIDE)
            {
                emissive = devicePrimitiveInstance.m_EmissiveColorAndScale * devicePrimitiveInstance.m_EmissiveColorAndScale.w;
            }

            assert(!isinf(emissive.x));
            assert(!isinf(emissive.y));
            assert(!isinf(emissive.z));
            assert(!isinf(emissive.w));
            assert(!isnan(emissive.x));
            assert(!isnan(emissive.y));
            assert(!isnan(emissive.z));
            assert(!isnan(emissive.w));

            //Multiply metallic and roughness with their scalar factors.
            const float metal = metalRoughness.z * material->m_MetallicFactor;
            const float roughness = metalRoughness.y * material->m_RoughnessFactor;

            //Take the local normal and tangent, then calculate the bitangent.
            float4 localNormal = make_float4(normalize(A->m_Normal * W + B->m_Normal * U + C->m_Normal * V), 0.f);
            float4 localTangent =
                (A->m_Tangent) * W +
                (B->m_Tangent) * U +
                (C->m_Tangent) * V;

            //const float flip = (A->m_Tangent.w + B->m_Tangent.w + C->m_Tangent.w) / 3.f;  //Is by definition the same for all vertices.
            localTangent = make_float4(normalize(make_float3(localTangent.x, localTangent.y, localTangent.z)), 0.f);

            auto& mat = devicePrimitiveInstance.m_Transform;

            //Transform to world space.
            float3 normalWorld = normalize(make_float3(mat * localNormal));
            float3 tangentWorld = normalize(make_float3(mat * localTangent));
            float3 bitangentWorld = cross(normalWorld, tangentWorld) * flip;

            //Extract the normal map normal, and then convert it to world space using the TBN matrix.
            float3 normalMapNormal = make_float3(normalMap.x, normalMap.y, normalMap.z);
            //normalMapNormal.y = 1.f - normalMapNormal.y;
            normalMapNormal = normalMapNormal * 2.f - 1.f;
            normalMapNormal = normalize(normalMapNormal);
            normalMapNormal = normalize(make_float3(
                normalMapNormal.x * tangentWorld.x + normalMapNormal.y * bitangentWorld.x + normalMapNormal.z * normalWorld.x,
                normalMapNormal.x * tangentWorld.y + normalMapNormal.y * bitangentWorld.y + normalMapNormal.z * normalWorld.y,
                normalMapNormal.x * tangentWorld.z + normalMapNormal.y * bitangentWorld.z + normalMapNormal.z * normalWorld.z
            ));//Matrix multiply except manually because I don't like sutil.

            //Can't have perfect specular surfaces. 0 is never acceptable.
            assert(roughness > 0.f && roughness <= 1.f);

            //ETA is air to surface.
            float eta = 1.f / material->m_IndexOfRefraction;

            //The surface data to write to. Local copy for fast access.
            SurfaceData output;
            output.m_PixelIndex = currIntersection.m_PixelIndex;
            output.m_IntersectionT = currIntersection.m_IntersectionT;
            output.m_Normal = normalMapNormal;
            output.m_GeometricNormal = normalWorld;
            output.m_Position = currRay.m_Origin + currRay.m_Direction * currIntersection.m_IntersectionT;
            output.m_IncomingRayDirection = currRay.m_Direction;
            output.m_TransportFactor = currRay.m_Contribution;
            output.m_Tangent = tangentWorld;

            auto& shadingData = output.m_ShadingData;
            shadingData.parameters = make_uint4(0u, 0u, 0u, 0u);    //Default init to 0 for all.

            //Set output color.
            output.m_ShadingData.color = finalColor;


            //Multiply factors with textures.
            const float4 clearCoat = tex2D<float4>(material->m_ClearCoatTexture, texCoords.x, texCoords.y);
            const float4 clearCoatRoughness = tex2D<float4>(material->m_ClearCoatRoughnessTexture, texCoords.x, texCoords.y);
            const float4 transmission = tex2D<float4>(material->m_TransmissionTexture, texCoords.x, texCoords.y);
            const float4 tint = tex2D<float4>(material->m_TintTexture, texCoords.x, texCoords.y);

            const float3 finalTintColor = make_float3(tint.x, tint.y, tint.z) * material->m_TintFactor;
            const float finalClearCoat = material->m_ClearCoatFactor * clearCoat.x;
            const float clearCoatGloss = 1.f - (material->m_ClearCoatRoughnessFactor * clearCoatRoughness.x);    //Invert from rough to gloss.
            const float finalTranmission = material->m_TransmissionFactor * transmission.x;

            shadingData.SetMetallic(metal);
            shadingData.SetRoughness(roughness);
            shadingData.SetSubSurface(material->m_SubSurfaceFactor);
            shadingData.SetSpecular(material->m_SpecularFactor);
            shadingData.SetSpecTint(material->m_SpecularTintFactor);
            shadingData.SetLuminance(material->m_Luminance);
            shadingData.SetAnisotropic(material->m_Anisotropic);
            shadingData.SetClearCoat(finalClearCoat);
            shadingData.SetClearCoatGloss(clearCoatGloss);
            shadingData.SetTint(finalTintColor);
            shadingData.SetSheen(material->m_SheenFactor);
            shadingData.SetSheenTint(material->m_SheenTintFactor);
            shadingData.SetTransmission(finalTranmission);
            shadingData.SetTransmittance(material->m_TransmittanceFactor);
            shadingData.SetETA(eta);

            
            //Useful debug print to show whether or not all information is correctly packed in the struct.
            /*
            if(material->m_IndexOfRefraction > 1.5f)
            {
                printf("Shading properties:\n eta: %f\n metallic: %f\n roughness: %f\n subsurface: %f\n specular: %f\n spectint: %f\n luminance: %f\n"\
                    " anisotropic: %f\n clearcoat: %f\n clearcoat: %f\n tint: %f %f %f\n sheen: %f\n sheentint: %f\n transmission: %f\n transmittance: %f %f %f\n"
                    , ETA, METALLIC, ROUGHNESS, SUBSURFACE, SPECULAR, SPECTINT, LUMINANCE, ANISOTROPIC, CLEARCOAT, CLEARCOATGLOSS, TINT.x, TINT.y, TINT.z, SHEEN, SHEENTINT, TRANSMISSION, shadingData.transmittance.x, shadingData.transmittance.y, shadingData.transmittance.z);
            }
            */
            

            output.m_Emissive = (emissive.x > 0 || emissive.y > 0 || emissive.z > 0);
            if (output.m_Emissive)
            {
                //Clamp between 0 and 1. TODO this is not HDR friendly so remove when we do that.
                output.m_ShadingData.color = make_float3(emissive);
                float maximum = fmaxf(output.m_ShadingData.color.x, fmaxf(output.m_ShadingData.color.y, output.m_ShadingData.color.z));
                output.m_ShadingData.color /= maximum;
            }

            a_OutPut[surfaceDataIndex] = output;
        }
    }
}