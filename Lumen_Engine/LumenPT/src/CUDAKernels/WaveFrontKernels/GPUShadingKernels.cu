#include "GPUShadingKernels.cuh"
#include <device_launch_parameters.h>

//Just temporary CUDA kernels.

CPU_ON_GPU void DEBUGShadePrimIntersections(
    const uint3 a_ResolutionAndDepth,
    const IntersectionRayBatch* const a_PrimaryRays,
    const IntersectionBuffer* const a_PrimaryIntersections,
    ResultBuffer* const a_Output)
{

    const unsigned int numPixels = a_ResolutionAndDepth.x * a_ResolutionAndDepth.y;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < numPixels; i += stride)
    {

        // Get intersection.
        const IntersectionData& currIntersection = a_PrimaryIntersections->GetIntersection(i, 0 /*There is only one ray per pixel, only one intersection per pixel (for now)*/);

        //printf("Pixel Index: %i, Total Pixels: %i \n", i, numPixels);

        if (currIntersection.IsIntersection())
        {

            // Get ray used to calculate intersection.
            const unsigned int rayArrayIndex = currIntersection.m_RayArrayIndex;

            const unsigned int vertexIndex = 3 * currIntersection.m_PrimitiveIndex;
            const DevicePrimitive* primitive = currIntersection.m_Primitive;

            //printf("vertex Index %i \n", vertexIndex);

            if (primitive == nullptr ||
                primitive->m_IndexBuffer == nullptr ||
                primitive->m_VertexBuffer == nullptr ||
                primitive->m_Material == nullptr)
            {

                if (!primitive)
                {
                    printf("Error, primitive: %p \n", primitive);
                }
                else
                {
                    printf("Error, found nullptr in primitive variables: \n\tm_IndexBuffer: %p \n\tm_VertexBuffer: %p \n\tm_Material: %p\n",
                           primitive->m_IndexBuffer,
                           primitive->m_VertexBuffer,
                           primitive->m_Material);
                }

                a_Output->SetPixel(make_float3(1.f, 0.f, 1.f), rayArrayIndex, ResultBuffer::OutputChannel::DIRECT);
                return;
            }

            /*printf("VertexIndex: %i, Primitive: %p, m_IndexBuffer: %p, m_VertexBuffer: %p \n",
                vertexIndex,
                primitive,
                primitive->m_IndexBuffer,
                primitive->m_VertexBuffer);*/

            const unsigned int vertexIndexA = primitive->m_IndexBuffer[vertexIndex + 0];
            const unsigned int vertexIndexB = primitive->m_IndexBuffer[vertexIndex + 1];
            const unsigned int vertexIndexC = primitive->m_IndexBuffer[vertexIndex + 2];

            //printf("VertexA Index: %i\n", vertexIndexA);
            //printf("VertexB Index: %i\n", vertexIndexB);
            //printf("VertexC Index: %i\n", vertexIndexC);

            const Vertex* A = &primitive->m_VertexBuffer[vertexIndexA];
            const Vertex* B = &primitive->m_VertexBuffer[vertexIndexB];
            const Vertex* C = &primitive->m_VertexBuffer[vertexIndexC];

            const float U = currIntersection.m_UVs.x;
            const float V = currIntersection.m_UVs.y;
            const float W = 1.f - (U + V);

            const float2 texCoords = A->m_UVCoord * W + B->m_UVCoord * U + C->m_UVCoord * V;

            if (U + V + W != 1.f)
            {
                printf("U: %f, V: %f, W: %f \n", U, V, W);
                a_Output->SetPixel(make_float3(1.f, 1.f, 0.f), rayArrayIndex, ResultBuffer::OutputChannel::DIRECT);
                return;
            }

            const DeviceMaterial* material = primitive->m_Material;

            const float4 textureColor = tex2D<float4>(material->m_DiffuseTexture, texCoords.x, texCoords.y);
            const float3 finalColor = make_float3(textureColor * material->m_DiffuseColor);

            a_Output->SetPixel(finalColor, rayArrayIndex, ResultBuffer::OutputChannel::DIRECT);
        }
    }
}


CPU_ON_GPU void WriteToOutput(
    const uint2 a_Resolution,
    const PixelBuffer* const a_Input,
    uchar4* a_Output)
{
    const unsigned int numPixels = a_Resolution.x * a_Resolution.y;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < numPixels; i += stride)
    {
        const auto color = make_color(a_Input->GetPixel(i, 0 /*Only one channel per pixel in the merged result*/));
        a_Output[i] = color;
    }
}