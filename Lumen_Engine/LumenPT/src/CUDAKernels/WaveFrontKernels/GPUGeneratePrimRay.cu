#include "GPUShadingKernels.cuh"
#include <device_launch_parameters.h>
#include "../../Shaders/CppCommon/SceneDataTableAccessor.h"

GPU_ONLY void HaltonSequence(
    unsigned int index,
    unsigned int base,
    float* result)
{
    ++index;

    float f = 1.f;
    float r = 0.f;

    while (index > 0)
    {
        f = f / base;
        r = r + f * (index % base);
        index = index / base;
    }

    *result = r;
}

CPU_ON_GPU void GeneratePrimaryRay(
    int a_NumRays,
    AtomicBuffer<IntersectionRayData>* const a_Buffer,
    float3 a_U,
    float3 a_V,
    float3 a_W,
    float3 a_Eye,
    uint2 a_Dimensions,
    unsigned int a_FrameCount)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < a_NumRays; i += stride)
    {
        //Convert the index into the screen dimensions.
        const int screenY = i / a_Dimensions.x;
        const int screenX = i - (screenY * a_Dimensions.x);

    	float2 jitter;
    	HaltonSequence(a_FrameCount + static_cast<unsigned int>(i), 2, &jitter.x);
    	HaltonSequence(a_FrameCount + static_cast<unsigned int>(i), 3, &jitter.y);
    	
        float3 direction = make_float3(static_cast<float>(screenX + jitter.x) / a_Dimensions.x,
                                       static_cast<float>(screenY + jitter.y) / a_Dimensions.y, 0.f);
        float3 origin = a_Eye;

        direction.x = -(direction.x * 2.0f - 1.0f);
        direction.y = -(direction.y * 2.0f - 1.0f);
        direction = normalize(direction.x * a_U + direction.y * a_V + a_W);

        IntersectionRayData ray{i, origin, direction, make_float3(1.f, 1.f, 1.f) };
        a_Buffer->Set(i, &ray); //Set because primary rays are ALWAYS a ray per pixel. No need to do atomic indexing. The atomic counter is manually set later.
    }
}

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
        //TODO: ensure that index is the same for the intersection data and ray.
        const IntersectionData currIntersection = *a_IntersectionData->GetData(i);
        const IntersectionRayData currRay = *a_Rays->GetData(currIntersection.m_RayArrayIndex);
        unsigned int surfaceDataIndex = currIntersection.m_PixelIndex;

        //TODO get material pointer from intersection data.
        //TODO extract barycentric coordinates and all that.
        //TODO store in output buffer.

        if (currIntersection.IsIntersection())
        {
            // Get ray used to calculate intersection.
            const unsigned int instanceId = currIntersection.m_InstanceId;
            const unsigned int rayArrayIndex = currIntersection.m_RayArrayIndex;
            const unsigned int vertexIndex = 3 * currIntersection.m_PrimitiveIndex;

            assert(a_SceneDataTable->GetTableEntry<DevicePrimitive>(instanceId) != nullptr);
            const auto devicePrimitive = *a_SceneDataTable->GetTableEntry<DevicePrimitive>(instanceId);

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
            const float metal = metalRoughness.z;
            const float roughness = metalRoughness.y;

            //Calculate the surface normal based on the texture and normal provided.

            //TODO: weigh based on barycentrics instead.
            float3 vertexNormal = normalize(A->m_Normal + B->m_Normal + C->m_Normal);


            //TODO base on barycentric instead
            //const float3 tangent = normalize(
            //    make_float3(
            //        A->m_Tangent.x + B->m_Tangent.x + C->m_Tangent.x,
            //        A->m_Tangent.y + B->m_Tangent.y + C->m_Tangent.y,
            //        A->m_Tangent.z + B->m_Tangent.z + C->m_Tangent.z
            //   ));
            const float3 tangent = normalize(make_float3(A->m_Tangent.x, A->m_Tangent.y, A->m_Tangent.z) * -A->m_Tangent.w);

            //Tangent nan? How?
            //assert(!isnan(length(tangent)));

            //tangent = normalize(tangent - dot(tangent, vertexNormal) * vertexNormal);

            float3 biTangent = normalize(cross(vertexNormal, tangent) * -A->m_Tangent.w);


            //TODO: This is in mesh space, not instance transformed space.
            //TODO: Get the transform of the mesh to get this to world space.
            sutil::Matrix3x3 tbn
            {
                tangent.x, biTangent.x, vertexNormal.x,
                tangent.y, biTangent.y, vertexNormal.y,
                tangent.z, biTangent.z, vertexNormal.z
            };

            ////Calculate the normal based on the vertex normal, tangent, bitangent and normal texture.
            //float3 actualNormal = make_float3(normalMap.x, normalMap.y, normalMap.z);
            //actualNormal = actualNormal * 2.f - 1.f;
            //actualNormal = normalize(actualNormal);
            //actualNormal = normalize(tbn * actualNormal);

            //assert(fabsf(length(actualNormal) - 1.f) < 0.001f);

            //Can't have perfect specular surfaces. 0 is never acceptable.
            assert(roughness > 0.f && roughness <= 1.f);

            //The surface data to write to. Local copy for fast access.
            SurfaceData output;
            output.m_Index = currIntersection.m_PixelIndex;
            output.m_Emissive = false; //TODO set based on whether or not a light was hit.
            output.m_IntersectionT = currIntersection.m_IntersectionT;
            output.m_Normal = vertexNormal; //TODO: use normal mapping.
            output.m_Color = finalColor;
            output.m_Position = currRay.m_Origin + currRay.m_Direction * currIntersection.m_IntersectionT;
            output.m_IncomingRayDirection = currRay.m_Direction;
            output.m_Metallic = metal;
            output.m_Roughness = roughness;
            output.m_TransportFactor = currRay.m_Contribution;

            a_OutPut[surfaceDataIndex] = output;
        }
    }
}
