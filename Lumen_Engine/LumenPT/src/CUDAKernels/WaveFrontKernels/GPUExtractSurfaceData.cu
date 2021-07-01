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
        const IntersectionRayData currRay = *a_Rays->GetData(i);
        const auto pixelIndex = currRay.m_PixelIndex;
        unsigned int surfaceDataIndex = PIXEL_DATA_INDEX(pixelIndex.m_X, pixelIndex.m_Y, a_Resolution.x);
    	
        if (currIntersection.IsIntersection())
        {
            // Get ray used to calculate intersection.
            const unsigned int instanceId = currIntersection.m_InstanceId;
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

            //Emissive mode
            float4 emissive = make_float4(0.f);

            //When enabled, just read the GLTF data and scale it accordingly.
            if (devicePrimitiveInstance.m_EmissionMode == Lumen::EmissionMode::ENABLED)
            {
                emissive = material->m_MaterialData.m_Emissive * devicePrimitiveInstance.m_EmissiveColorAndScale.w;
                emissive *= tex2D<float4>(material->m_EmissiveTexture, texCoords.x, texCoords.y);
            }

            //When override, take the ovverride emissive color and scale it up.
            else if (devicePrimitiveInstance.m_EmissionMode == Lumen::EmissionMode::OVERRIDE)
            {
                emissive = devicePrimitiveInstance.m_EmissiveColorAndScale * devicePrimitiveInstance.m_EmissiveColorAndScale.w;
            }

            assert(!isinf(emissive.x));
            assert(!isinf(emissive.y));
            assert(!isinf(emissive.z));
            //assert(!isinf(emissive.w));
            assert(!isnan(emissive.x));
            assert(!isnan(emissive.y));
            assert(!isnan(emissive.z));
            //assert(!isnan(emissive.w));

            //The surface data to write to. Local copy for fast access.
            SurfaceData output;
            output.m_SurfaceFlags = SURFACE_FLAG_NONE;  //Default to no flags.

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
            const float3 normalWorld = normalize(make_float3(mat * localNormal));
            const float3 tangentWorld = normalize(make_float3(mat * localTangent));
            const float3 bitangentWorld = cross(normalWorld, tangentWorld) * flip;

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

            //If emissive, set flag and 
            if ((emissive.x > 0.f || emissive.y > 0.f || emissive.z > 0.f))
            {
                //Clamp between 0 and 1. TODO this is not HDR friendly so remove when we do that.
                output.m_MaterialData.m_Color = emissive;
                float maximum = fmaxf(output.m_MaterialData.m_Color.x, fmaxf(output.m_MaterialData.m_Color.y, output.m_MaterialData.m_Color.z));
                output.m_MaterialData.m_Color /= maximum;
            	
                //Set the output flag. Because this surface is emissive, it will never be shaded and paths are always terminated.
                //NOTE: Surface data position is not set, neither is normal. This means that it is not safe to do anything on an emissive surface that is randomly hit.
                output.m_SurfaceFlags |= SURFACE_FLAG_EMISSIVE;
                output.m_PixelIndex = pixelIndex;
                output.m_IntersectionT = currIntersection.m_IntersectionT;
                output.m_Normal = normalMapNormal;
                a_OutPut[surfaceDataIndex] = output;

                continue;
            }

            //Check for alpha discard (discard anything that is somewhat transparent).
            if (textureColor.w < 0.51f)
            {
                //The position of intersection is set so that the ray can continue onward.
                output.m_SurfaceFlags |= SURFACE_FLAG_ALPHA_TRANSPARENT;
                output.m_Position = currRay.m_Origin + currRay.m_Direction * currIntersection.m_IntersectionT;
                output.m_IncomingRayDirection = currRay.m_Direction;
                output.m_TransportFactor = currRay.m_Contribution;
                output.m_PixelIndex = pixelIndex;
                output.m_IntersectionT = currIntersection.m_IntersectionT;
                output.m_Normal = normalMapNormal;
                a_OutPut[surfaceDataIndex] = output;
                continue;
            }

            //ETA is air to surface.
            float eta = 1.f / material->m_MaterialData.GetRefractiveIndex();

            output.m_PixelIndex = pixelIndex;
            output.m_IntersectionT = currIntersection.m_IntersectionT;
            output.m_Normal = normalMapNormal;
            output.m_GeometricNormal = normalWorld;
            output.m_Position = currRay.m_Origin + currRay.m_Direction * currIntersection.m_IntersectionT;
            output.m_IncomingRayDirection = currRay.m_Direction;
            output.m_TransportFactor = currRay.m_Contribution;
            output.m_Tangent = tangentWorld;

        	//Copy the material properties over.
            output.m_MaterialData = material->m_MaterialData;

            //Multiply metallic and roughness with their scalar factors. Write to output.
            const float4 metalRoughness = tex2D<float4>(material->m_MetalRoughnessTexture, texCoords.x, texCoords.y);
            output.m_MaterialData.SetMetallic(metalRoughness.z * material->m_MaterialData.GetMetallic());
            output.m_MaterialData.SetRoughness(metalRoughness.y * material->m_MaterialData.GetRoughness());
        	
            //Set output color.
            const float4 finalColor = textureColor * material->m_MaterialData.m_Color;
            output.m_MaterialData.m_Color = finalColor;

            //Multiply factors with textures.
            const float4 clearCoat = tex2D<float4>(material->m_ClearCoatTexture, texCoords.x, texCoords.y);
            const float4 clearCoatRoughness = tex2D<float4>(material->m_ClearCoatRoughnessTexture, texCoords.x, texCoords.y);
            const float4 transmission = tex2D<float4>(material->m_TransmissionTexture, texCoords.x, texCoords.y);
            const float4 tint = tex2D<float4>(material->m_TintTexture, texCoords.x, texCoords.y);

            const float3 finalTintColor = make_float3(tint.x, tint.y, tint.z) * material->m_MaterialData.GetTint();
            const float finalClearCoat = material->m_MaterialData.GetClearCoat() * clearCoat.x;
            const float clearCoatGloss = material->m_MaterialData.GetClearCoatGloss() * (1.f - clearCoatRoughness.x); //Invert the texture because it's roughness and not gloss.
            const float finalTranmission = material->m_MaterialData.GetTransmission() * transmission.x;

            //Can't have perfect specular surfaces. 0 is never acceptable.
            assert(output.m_MaterialData.GetRoughness() > 0.f && output.m_MaterialData.GetRoughness() <= 1.f);
        	
            output.m_MaterialData.SetClearCoat(finalClearCoat);
            output.m_MaterialData.SetClearCoatGloss(clearCoatGloss);
            output.m_MaterialData.SetTint(finalTintColor);
            output.m_MaterialData.SetTransmission(finalTranmission);
            output.m_MaterialData.SetRefractiveIndex(eta);


        	//////TODO Comment this out. Debugging only.
            //if(i == 220200)
            //{
            //    printf("Surface data material values:\n");
            //    printf("- Color: %f %f %f %f\n", output.m_MaterialData.m_Color.x, output.m_MaterialData.m_Color.y, output.m_MaterialData.m_Color.z, output.m_MaterialData.m_Color.w);
            //    printf("- Emissive: %f %f %f %f\n", output.m_MaterialData.m_Emissive.x, output.m_MaterialData.m_Emissive.y, output.m_MaterialData.m_Emissive.z, output.m_MaterialData.m_Emissive.w);
            //    printf("- Tint: %f %f %f\n", output.m_MaterialData.m_Tint.x, output.m_MaterialData.m_Tint.y, output.m_MaterialData.m_Tint.z);
            //    printf("- Transmittance: %f %f %f\n", output.m_MaterialData.m_Transmittance.x, output.m_MaterialData.m_Transmittance.y, output.m_MaterialData.m_Transmittance.z);
            //    printf("- IOR: %f\n", output.m_MaterialData.GetRefractiveIndex());
            //    printf("- Spec: %f\n", output.m_MaterialData.GetSpecular());
            //    printf("- SpecTint: %f\n", output.m_MaterialData.GetSpecTint());
            //    printf("- Luminance: %f\n", output.m_MaterialData.GetLuminance());
            //    printf("- Metallic: %f\n", output.m_MaterialData.GetMetallic());
            //    printf("- Roughness: %f\n", output.m_MaterialData.GetRoughness());
            //    printf("- SubSurface: %f\n", output.m_MaterialData.GetSubSurface());
            //    printf("- Anisotropic: %f\n", output.m_MaterialData.GetAnisotropic());
            //    printf("- Sheen: %f\n", output.m_MaterialData.GetSheen());
            //    printf("- SheenTint: %f\n", output.m_MaterialData.GetSheenTint());
            //    printf("- ClearCoat: %f\n", output.m_MaterialData.GetClearCoat());
            //    printf("- Transmission: %f\n", output.m_MaterialData.GetTransmission());
            //}
        	
            a_OutPut[surfaceDataIndex] = output;
        }
        else
        {
        	//No intersection, so mark with the non-intersection bit.
			a_OutPut[surfaceDataIndex].m_SurfaceFlags = SURFACE_FLAG_NON_INTERSECT;
        }
    }
}