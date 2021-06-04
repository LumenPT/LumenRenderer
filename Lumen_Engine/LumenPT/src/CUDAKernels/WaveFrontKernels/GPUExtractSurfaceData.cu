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
    SceneDataTableAccessor* a_SceneDataTable)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < a_NumIntersections; i += stride)
    {
        const IntersectionData currIntersection = *a_IntersectionData->GetData(i);
        const IntersectionRayData currRay = *a_Rays->GetData(currIntersection.m_RayArrayIndex);
        unsigned int surfaceDataIndex = currIntersection.m_PixelIndex;

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

            const float2 texCoords = A->m_UVCoord * W + B->m_UVCoord * U + C->m_UVCoord * V;

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
            
            //Calculate the surface normal based on the texture and normal provided.

            //TODO: weigh based on barycentrics instead.
            float3 surfaceNormal = normalize(A->m_Normal * W + B->m_Normal * U + C->m_Normal * V);

            //Transform the normal to world space
            float4 surfaceNormalWorld = devicePrimitiveInstance.m_Transform * make_float4(surfaceNormal, 0.f);


            //TODO normal mapping
            float3 tangent = 
                normalize(
                    (make_float3(A->m_Tangent) * A->m_Tangent.w) * W + 
                    (make_float3(B->m_Tangent) * B->m_Tangent.w) * U + 
                    (make_float3(C->m_Tangent) * C->m_Tangent.w) * V
                );
            assert(!isnan(length(tangent)));
            assert(length(tangent) > 0);

            ////tangent = normalize(tangent);
            ////assert(!isnan(length(tangent)));

            //float3 biTangent = normalize(cross(vertexNormal, tangent));

            //sutil::Matrix3x3 tbn
            //{
            //    tangent.x, biTangent.x, vertexNormal.x,
            //    tangent.y, biTangent.y, vertexNormal.y,
            //    tangent.z, biTangent.z, vertexNormal.z
            //};

            //////Calculate the normal based on the vertex normal, tangent, bitangent and normal texture.
            //float3 actualNormal = make_float3(normalMap.x, normalMap.y, normalMap.z);
            //actualNormal = actualNormal * 2.f - 1.f;
            //actualNormal = normalize(actualNormal);
            //actualNormal = tbn * actualNormal;
            ////Transform the normal to world space (0 for no translation).
            //auto normalWorldSpace = devicePrimitiveInstance.m_Transform * make_float4(actualNormal, 0.f);
            //actualNormal = normalize(make_float3(normalWorldSpace));
            //assert(fabsf(length(actualNormal) - 1.f) < 0.01f);

            //Can't have perfect specular surfaces. 0 is never acceptable.
            assert(roughness > 0.f && roughness <= 1.f);

            //The final normal to be used by all shading.
            float3 normal = normalize(make_float3(surfaceNormalWorld));

            //ETA is air to surface.
            float eta = 1.f / material->m_IndexOfRefraction;

            //The surface data to write to. Local copy for fast access.
            SurfaceData output;
            output.m_Index = currIntersection.m_PixelIndex;
            output.m_IntersectionT = currIntersection.m_IntersectionT;
            output.m_Normal = normal;
            output.m_Position = currRay.m_Origin + currRay.m_Direction * currIntersection.m_IntersectionT;
            output.m_IncomingRayDirection = currRay.m_Direction;
            output.m_TransportFactor = currRay.m_Contribution;
            output.m_Tangent = tangent;

            auto& shadingData = output.m_ShadingData;
            shadingData.parameters = make_uint4(0u, 0u, 0u, 0u);    //Default init to 0 for all.

            //Set output color.
            output.m_ShadingData.color = finalColor;

            //TODO enable this.
            if(true)
            {
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
            }
            else 
            {
                //TODO remove
                shadingData.SetMetallic(0.f);
                shadingData.SetRoughness(1.f);
                shadingData.SetSubSurface(0.f);
                shadingData.SetSpecular(0.f);
                shadingData.SetSpecTint(0.f);
                shadingData.SetLuminance(1.f);
                shadingData.SetAnisotropic(0.f);
                shadingData.SetClearCoat(0.f);
                shadingData.SetClearCoatGloss(0.f);
                shadingData.SetTint(make_float3(1.f, 1.f, 1.f));
                shadingData.SetSheen(0.f);
                shadingData.SetSheenTint(0.f);
                shadingData.SetTransmission(0.f);
                shadingData.SetTransmittance(make_float3(0.9f, 0.9f, 0.9f));
                shadingData.SetETA(1.f);
            }

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