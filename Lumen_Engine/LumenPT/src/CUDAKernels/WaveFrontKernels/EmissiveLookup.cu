#include "EmissiveLookup.cuh"
//#include "../../Framework/CudaUtilities.h"
#include "../../Framework/PTMaterial.h"
#include <sutil/vec_math.h>
#include <cuda_runtime.h>

CPU_ON_GPU void FindEmissives(const Vertex* a_Vertices, bool* a_EmissiveBools, const uint32_t* a_Indices, const DeviceMaterial* a_Mat, const uint8_t a_VertexBufferSize)
{
    //const auto devMat = a_Mat->GetDeviceMaterial();

    //pack these into triangle
    //find texture coordinates on this triangle (rather than just vertices
    //sample texture at area of triangle through UVs

    for (unsigned int i = 0; i < a_VertexBufferSize; i++)
    {
        if (i % 3 == 0) 
        {
            //looped over 3 vertices, construct triangle
            const Vertex& vert0 = a_Vertices[i - 2];
            const Vertex& vert1 = a_Vertices[i - 1];
            const Vertex& vert2 = a_Vertices[i];
            
            //calculate triangle area using Heron's formula
            //const float a = sqrtf(
            //    powf((vert0.m_UVCoord.y - vert0.m_UVCoord.x), 2) *
            //    powf((vert1.m_UVCoord.y - vert1.m_UVCoord.x), 2));
            //const float b = sqrtf(
            //    powf((vert1.m_UVCoord.y - vert1.m_UVCoord.x), 2) *
            //    powf((vert2.m_UVCoord.y - vert2.m_UVCoord.x), 2));
            //const float c = sqrtf(
            //    powf((vert2.m_UVCoord.y - vert2.m_UVCoord.x), 2) *
            //    powf((vert0.m_UVCoord.y - vert0.m_UVCoord.x), 2));
            //
            //const float semiPerim = 0.5f * (a + b + c);
            //const float texCoordArea = sqrtf(semiPerim *
            //    (semiPerim - a) *
            //    (semiPerim - b) *
            //    (semiPerim - c));

            const float2 UVCentroid = make_float2(
                (vert0.m_UVCoord.x + vert1.m_UVCoord.x + vert2.m_UVCoord.x) / 3, 
                (vert0.m_UVCoord.y + vert1.m_UVCoord.y + vert2.m_UVCoord.y) / 3);

            //sample emission at UVCentroid
            float4 diffCol = tex2D<float4>(a_Mat->m_DiffuseTexture, UVCentroid.x, UVCentroid.y);
            float4 emissCol = tex2D<float4>(a_Mat->m_EmissiveTexture, UVCentroid.x, UVCentroid.y);
            float4 chanEmissCol = make_float4(a_Mat->m_EmissionColor.x, a_Mat->m_EmissionColor.y, a_Mat->m_EmissionColor.z, 1.0f);
            emissCol *= chanEmissCol;

            float3 emission = make_float3(diffCol.x * emissCol.x, diffCol.y * emissCol.y, diffCol.z * emissCol.z);
            float3 nullVec = make_float3(0, 0, 0);

            //if diffuse is not transparent and emission is not 0
            if (diffCol.w != 0.0f && emission.x != 0.0f && emission.y != 0.0f && emission.z != 0.0f)
            {
                a_EmissiveBools[i] = true;
                continue;
            }

            a_EmissiveBools[i] = false;
            continue;
        }

        //const Vertex& vertex = a_Vertices[i];
        //
        //float4 diffCol = tex2D<float4>(devMat->m_DiffuseTexture, vertex.m_UVCoord.x, vertex.m_UVCoord.y);
        //float3 emissCol = tex2D<float3>(devMat->m_EmissiveTexture, vertex.m_UVCoord.x, vertex.m_UVCoord.y);

        //multiply emissive tex with emissive color
        //will also need indices

    }
}