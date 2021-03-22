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
    int2 a_Dimensions,
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
        const IntersectionData* currIntersection = a_IntersectionData->GetData(i);
        const IntersectionRayData* currRay = a_Rays->GetData(currIntersection->m_RayArrayIndex);
        unsigned int surfaceDataIndex = currIntersection->m_PixelIndex;

        //TODO get material pointer from intersection data.
        //TODO extract barycentric coordinates and all that.
        //TODO store in output buffer.


        if (currIntersection->IsIntersection())
        {
            // Get ray used to calculate intersection.
            const unsigned int instanceId = currIntersection->m_InstanceId;
            const unsigned int rayArrayIndex = currIntersection->m_RayArrayIndex;
            const unsigned int vertexIndex = 3 * currIntersection->m_PrimitiveIndex;
            auto devicePrimitive = a_SceneDataTable->GetTableEntry<DevicePrimitive>(instanceId);

            if (devicePrimitive == nullptr ||
                devicePrimitive->m_IndexBuffer == nullptr ||
                devicePrimitive->m_VertexBuffer == nullptr ||
                devicePrimitive->m_Material == nullptr)
            {

                if (!devicePrimitive)
                {
                    printf("Error, primitive: %p \n", devicePrimitive);
                }
                else
                {
                    printf("Error, found nullptr in primitive variables: \n\tm_IndexBuffer: %p \n\tm_VertexBuffer: %p \n\tm_Material: %p\n",
                           devicePrimitive->m_IndexBuffer,
                           devicePrimitive->m_VertexBuffer,
                           devicePrimitive->m_Material);
                }

                //Set to debug color purple.
                a_OutPut[surfaceDataIndex].m_Color = make_float3(1.f, 0.f, 1.f);
                return;
            }

            /*printf("VertexIndex: %i, Primitive: %p, m_IndexBuffer: %p, m_VertexBuffer: %p \n",
                vertexIndex,
                primitive,
                primitive->m_IndexBuffer,
                primitive->m_VertexBuffer);*/

            const unsigned int vertexIndexA = devicePrimitive->m_IndexBuffer[vertexIndex + 0];
            const unsigned int vertexIndexB = devicePrimitive->m_IndexBuffer[vertexIndex + 1];
            const unsigned int vertexIndexC = devicePrimitive->m_IndexBuffer[vertexIndex + 2];

            //printf("VertexA Index: %i\n", vertexIndexA);
            //printf("VertexB Index: %i\n", vertexIndexB);
            //printf("VertexC Index: %i\n", vertexIndexC);

            const Vertex* A = &devicePrimitive->m_VertexBuffer[vertexIndexA];
            const Vertex* B = &devicePrimitive->m_VertexBuffer[vertexIndexB];
            const Vertex* C = &devicePrimitive->m_VertexBuffer[vertexIndexC];

            const float U = currIntersection->m_Barycentrics.x;
            const float V = currIntersection->m_Barycentrics.y;
            const float W = 1.f - (U + V);

            const float2 texCoords = A->m_UVCoord * W + B->m_UVCoord * U + C->m_UVCoord * V;

            const DeviceMaterial* material = devicePrimitive->m_Material;

            //TODO extract different textures (emissive, diffuse, metallic, roughness).
            const float4 textureColor = tex2D<float4>(material->m_DiffuseTexture, texCoords.x, texCoords.y);
            const float3 finalColor = make_float3(textureColor * material->m_DiffuseColor);

            //The surface data to write to.
            auto* output = &a_OutPut[surfaceDataIndex];

            //TODO set all these.
            output->m_Index = currIntersection->m_PixelIndex;
            output->m_Emissive = false; //TODO
            output->m_Color = finalColor;
            output->m_IntersectionT = currIntersection->m_IntersectionT;
            output->m_Normal = normalize(A->m_Normal + B->m_Normal + C->m_Normal);  //TODO untested.
            output->m_Position = currRay->m_Origin + currRay->m_Direction * currIntersection->m_IntersectionT;
            output->m_IncomingRayDirection = currRay->m_Direction;
            output->m_Metallic = 1;     //TODO
            output->m_Roughness = 1;    //TODO
            output->m_TransportFactor = currRay->m_Contribution;
        
        }
        
    }

}
