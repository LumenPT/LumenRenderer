#include "GPUShadingKernels.cuh"
#include "../../Shaders/CppCommon/SceneDataTableAccessor.h"
#include <Lumen/ModelLoading/MeshInstance.h>
#include <device_launch_parameters.h>
#include <sutil/vec_math.h>
#include <sutil/Matrix.h>


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
            const float4 mappedNormal = tex2D<float4>(material->m_NormalTexture, texCoords.x, texCoords.y);
            const float4 textureColor = tex2D<float4>(material->m_DiffuseTexture, texCoords.x, texCoords.y);
            const float3 surfaceColor = make_float3(textureColor * material->m_DiffuseColor);
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


            const float metal = metalRoughness.z;
            const float roughness = metalRoughness.y;
            
            //Calculate the surface normal based on the texture and normal provided.
            const float3 surfaceNormal = 
                A->m_Normal * W + 
                B->m_Normal * U + 
                C->m_Normal * V;

            //TODO normal mapping
            const float3 surfaceTangent = 
                make_float3(A->m_Tangent) * W + 
                make_float3(B->m_Tangent) * U + 
                make_float3(C->m_Tangent) * V;

            //assert(!isnan(length(tangent)));
            //assert(length(tangent) > 0);

            ////tangent = normalize(tangent);
            ////assert(!isnan(length(tangent)));
            
            //According to the gltf2.0 spec, all the tangents' w components of each vertex in a triangle should be the same.
            assert(A->m_Tangent.w == B->m_Tangent.w == C->m_Tangent.w);
            const float tangentSpaceHandedness = A->m_Tangent.w;

            const float3 surfaceBiTangent = cross(surfaceNormal, surfaceTangent) * tangentSpaceHandedness;

            //Transform the normal, tangent and bi-tangent to world space
            const float3 surfaceNormalWorld = normalize(make_float3(devicePrimitiveInstance.m_Transform * make_float4(surfaceNormal, 0.f)));
            const float3 surfaceTangentWorld = normalize(make_float3(devicePrimitiveInstance.m_Transform * make_float4(surfaceTangent, 0.f)));
            const float3 surfaceBiTangentWorld = normalize(make_float3(devicePrimitiveInstance.m_Transform * make_float4(surfaceBiTangent, 0.f)));

            sutil::Matrix3x3 tbn
            {
                surfaceTangentWorld.x, surfaceBiTangentWorld.x, surfaceNormalWorld.x,
                surfaceTangentWorld.y, surfaceBiTangentWorld.y, surfaceNormalWorld.y,
                surfaceTangentWorld.z, surfaceBiTangentWorld.z, surfaceNormalWorld.z
            };

            //////Calculate the normal based on the vertex normal, tangent, bitangent and normal texture.
            float3 mappedNormalWorld = make_float3(mappedNormal) * 2.f - 1.f;
            mappedNormalWorld = normalize(tbn * mappedNormalWorld);
            ////Transform the normal to world space (0 for no translation).
            //auto normalWorldSpace = devicePrimitiveInstance.m_Transform * make_float4(actualNormal, 0.f);
            //actualNormal = normalize(make_float3(normalWorldSpace));
            //assert(fabsf(length(actualNormal) - 1.f) < 0.01f);

            //Can't have perfect specular surfaces. 0 is never acceptable.
            assert(roughness > 0.f && roughness <= 1.f);

            //The surface data to write to. Local copy for fast access.
            SurfaceData output;
            output.m_Index = currIntersection.m_PixelIndex;
            output.m_IntersectionT = currIntersection.m_IntersectionT;
            output.m_Normal = normalize(surfaceNormalWorld);
            output.m_Color = surfaceColor;
            output.m_Position = currRay.m_Origin + currRay.m_Direction * currIntersection.m_IntersectionT;
            output.m_IncomingRayDirection = currRay.m_Direction;
            output.m_Metallic = metal;
            output.m_Roughness = roughness;
            output.m_TransportFactor = currRay.m_Contribution;
            output.m_Emissive = (emissive.x > 0 || emissive.y > 0 || emissive.z > 0);
            if (output.m_Emissive)
            {
                //Clamp between 0 and 1. TODO this is not HDR friendly so remove when we do that.
                output.m_Color = make_float3(emissive);
                float max = fmaxf(output.m_Color.x, fmaxf(output.m_Color.y, output.m_Color.z));
                output.m_Color /= max;
            }

            a_OutPut[surfaceDataIndex] = output;
        }
    }
}