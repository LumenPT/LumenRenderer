#include "EmissiveLookup.cuh"
#include "../../Framework/CudaUtilities.h"
#include "../../Framework/PTMaterial.h"
#include <sutil/vec_math.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda/device_atomic_functions.h>
#include <cassert>


CPU_ONLY void FindEmissivesWrap(
    const Vertex* a_Vertices,
    const uint32_t* a_Indices,
    bool* a_Emissives,
    const DeviceMaterial* a_Mat,
    const uint8_t a_IndexBufferSize,
    unsigned int& a_NumLights)
{
    unsigned int* numLightsPtr;
    cudaMalloc(&numLightsPtr, sizeof(unsigned int));

    FindEmissives <<<1, 1>>> (a_Vertices, a_Indices, a_Emissives, a_Mat, a_IndexBufferSize, numLightsPtr);
    cudaDeviceSynchronize();
    cudaMemcpy(&a_NumLights, numLightsPtr, sizeof(unsigned int), cudaMemcpyDeviceToHost);
}

CPU_ON_GPU void FindEmissives(
    const Vertex* a_Vertices,
    const uint32_t* a_Indices,
    bool* a_Emissives,
    const DeviceMaterial* a_Mat,
    const uint8_t a_IndexBufferSize,
    unsigned int* a_NumLights)
{
    //const auto devMat = a_Mat->GetDeviceMaterial();

    //pack these into triangle
    //find texture coordinates on this triangle (rather than just vertices
    //sample texture at area of triangle through UVs

    for (unsigned int baseIndex = 0; baseIndex < a_IndexBufferSize; baseIndex+=3)
    {

            //looped over 3 vertices, construct triangle
            const Vertex& vert0 = a_Vertices[baseIndex + 0];
            const Vertex& vert1 = a_Vertices[baseIndex + 1];
            const Vertex& vert2 = a_Vertices[baseIndex + 2];
            
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

            //if diffuse is not transparent and emission is not 0
            if (diffCol.w != 0.0f && emission.x != 0.0f && emission.y != 0.0f && emission.z != 0.0f)
            {
                a_Emissives[baseIndex / 3] = true;
                a_NumLights++;
                continue;
            }

            a_Emissives[baseIndex / 3] = false;

        //const Vertex& vertex = a_Vertices[i];
        //
        //float4 diffCol = tex2D<float4>(devMat->m_DiffuseTexture, vertex.m_UVCoord.x, vertex.m_UVCoord.y);
        //float3 emissCol = tex2D<float3>(devMat->m_EmissiveTexture, vertex.m_UVCoord.x, vertex.m_UVCoord.y);

        //multiply emissive tex with emissive color
        //will also need indices

    }
}

CPU_ONLY void AddToLightBufferWrap(
    const Vertex* a_Vertices,
    const uint32_t* a_Indices,
    const bool* a_Emissives,
    const uint8_t a_IndexBufferSize,
    WaveFront::AtomicBuffer<WaveFront::TriangleLight>* a_Lights,
    sutil::Matrix4x4 a_TransformMat)
{
    AddToLightBuffer <<<1, 1>>> (a_Vertices, a_Indices, a_Emissives, a_IndexBufferSize, a_Lights, a_TransformMat);
}

CPU_ON_GPU void AddToLightBuffer(
    const Vertex* a_Vertices,
    const uint32_t* a_Indices,
    const bool* a_Emissives,
    const uint8_t a_IndexBufferSize,
    WaveFront::AtomicBuffer<WaveFront::TriangleLight>* a_Lights,
    sutil::Matrix4x4 a_TransformMat)
{
    //assert(sizeof(a_Vertices) <= 0);

    printf("Constructing lights buffer\n");


    //Loop over triangles in primitive
    for (unsigned int baseIndex = 0; baseIndex < a_IndexBufferSize; baseIndex+=3)
    {

        //check first vertex of triangle to see if its in emissive buffer
        if (a_Emissives[baseIndex / 3] == true)
        //if (i % 3 == 0 && a_Emissives[i - 2] == true)
        {
            const Vertex& vert0 = a_Vertices[baseIndex + 0];
            const Vertex& vert1 = a_Vertices[baseIndex + 1];
            const Vertex& vert2 = a_Vertices[baseIndex + 2];

            //check if normal is here
            float4 tempWorldPos;
            WaveFront::TriangleLight* light;
            //Need it this way because need float3 to float4 and back conversions
            //Light vertex 0
            tempWorldPos = { vert0.m_Position.x, vert0.m_Position.y, vert0.m_Position.z, 0.0f };
            tempWorldPos = a_TransformMat * tempWorldPos;
            light->p0 = { tempWorldPos.x, tempWorldPos.y, tempWorldPos.z };
            //Light vertex 1
            tempWorldPos = { vert1.m_Position.x, vert1.m_Position.y, vert1.m_Position.z, 0.0f };
            tempWorldPos = a_TransformMat * tempWorldPos;
            light->p1 = { tempWorldPos.x, tempWorldPos.y, tempWorldPos.z };
            //Light vertex 2
            tempWorldPos = { vert2.m_Position.x, vert2.m_Position.y, vert2.m_Position.z, 0.0f };
            tempWorldPos = a_TransformMat * tempWorldPos;
            light->p2 = { tempWorldPos.x, tempWorldPos.y, tempWorldPos.z };

            light->radiance = {2000, 2000, 2000};
            light->normal = (vert0.m_Normal + vert1.m_Normal + vert2.m_Normal) * 0.33f;

            float3 vec1 = light->p0 - light->p1;
            float3 vec2 = light->p0 - light->p2;
            
            light->area = sqrtf(
                pow((vec1.y * vec2.z - vec2.y * vec1.z), 2) +
                pow((vec1.x * vec2.z - vec2.x * vec1.z), 2) +
                pow((vec1.x * vec2.y - vec2.x * vec1.y), 2)
            ) / 2.0f;

            a_Lights->Add(light);

            //Add light to lightbuffer, but to know where the end of the buffer is, keep track of lightIndex.
            //Dont know how many lights have already been added to buffer.
        }
    }
    //cudaDeviceSynchronize();

}

//CPU_ON_GPU void AddToLightBuffer2()
//{
//    /*WaveFront::TriangleLight* light;
//
//    light->p0 = { 75.f, 75.f, 75.f };
//    light->p1 = { 100.f,75.f, 100.f };
//    light->p2 = { 25.f, 100.f, 25.f };
//
//    light->radiance = { 2000, 2000, 2000 };
//    light->normal = {0.f, -1.f, 0.f};
//
//    float3 vec1 = light->p0 - light->p1;
//    float3 vec2 = light->p0 - light->p2;
//
//    light->area = sqrtf(
//        pow((vec1.y * vec2.z - vec2.y * vec1.z), 2) +
//        pow((vec1.x * vec2.z - vec2.x * vec1.z), 2) +
//        pow((vec1.x * vec2.y - vec2.x * vec1.y), 2)
//    ) / 2.0f;
//
//    light->area = sqrtf(
//        pow((vec1.y * vec2.z - vec2.y * vec1.z), 2) +
//        pow((vec1.x * vec2.z - vec2.x * vec1.z), 2) +
//        pow((vec1.x * vec2.y - vec2.x * vec1.y), 2)
//    ) / 2.0f;
//
//    a_Lights->Add(light);*/
//    cudaDeviceSynchronize();
//    printf("calling AddToLightBuffer2\n");
//
//}