#include "GPUEmissiveLookup.cuh"
#include "../../Framework/CudaUtilities.h"
#include "../../Framework/PTMaterial.h"
#include <sutil/vec_math.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <device_launch_parameters.h>
#include "../../Shaders/CppCommon/SceneDataTableAccessor.h"
#include <Lumen/ModelLoading/MeshInstance.h>


CPU_ON_GPU void FindEmissivesGpu(
    const Vertex* a_Vertices,
    const uint32_t* a_Indices,
    bool* a_Emissives,
    const DeviceMaterial* a_Mat,
    const uint32_t a_IndexBufferSize,
    unsigned int* a_NumLights)
{
	//Set to 0.
    *a_NumLights = 0;
	
    //const auto devMat = a_Mat->GetDeviceMaterial();

    //pack these into triangle
    //find texture coordinates on this triangle (rather than just vertices
    //sample texture at area of triangle through UVs

    for (unsigned int baseIndex = 0; baseIndex < a_IndexBufferSize; baseIndex+=3)
    {
        //looped over 3 vertices, construct triangle

        const unsigned index0 = a_Indices[baseIndex + 0];
        const unsigned index1 = a_Indices[baseIndex + 1];
        const unsigned index2 = a_Indices[baseIndex + 2];

        const Vertex& vert0 = a_Vertices[index0];
        const Vertex& vert1 = a_Vertices[index1];
        const Vertex& vert2 = a_Vertices[index2];
        
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

        constexpr float oneThird = 1.f / 3.f;

        const float2 UVCentroid = (vert0.m_UVCoord + vert1.m_UVCoord + vert2.m_UVCoord) * oneThird;

        //auto diffuseTexture = a_Mat->m_DiffuseTexture;
        auto emissiveTexture = a_Mat->m_EmissiveTexture;

        //float4 diffuseColor = a_Mat->m_DiffuseColor;

        //if(diffuseTexture)
        //{
        //    diffuseColor *= tex2D<float4>(diffuseTexture, UVCentroid.x, UVCentroid.y);
        //}

        float4 emissiveColor = a_Mat->m_MaterialData.m_Emissive;

        if(emissiveTexture)
        {
            //sample emission at UVCentroid
            const float4 emissiveTextureColor = tex2D<float4>(emissiveTexture, UVCentroid.x, UVCentroid.y);
            emissiveColor *= emissiveTextureColor;
        }

        const float4 finalEmission = emissiveColor;

        assert(!isnan(finalEmission.x));
        assert(!isnan(finalEmission.y));
        assert(!isnan(finalEmission.z));
        assert(!isnan(finalEmission.w));

        const unsigned triangleIndex = baseIndex / 3; //Base index goes up by three each loop, divide by three to get the num of triangles before current value.
        //if emission not equal to 0
        if ((finalEmission.x > 0.0f || finalEmission.y > 0.0f || finalEmission.z > 0.0f))
        {        	
            a_Emissives[triangleIndex] = true;
            (*a_NumLights)++;
            continue;
        }

        a_Emissives[triangleIndex] = false;

        //const Vertex& vertex = a_Vertices[i];
        //
        //float4 diffCol = tex2D<float4>(devMat->m_DiffuseTexture, vertex.m_UVCoord.x, vertex.m_UVCoord.y);
        //float3 emissCol = tex2D<float3>(devMat->m_EmissiveTexture, vertex.m_UVCoord.x, vertex.m_UVCoord.y);

        //multiply emissive tex with emissive color
        //will also need indices

    }
}



CPU_ON_GPU void AddToLightBufferGpu(
    const uint32_t a_IndexBufferSize,
    WaveFront::AtomicBuffer<WaveFront::TriangleLight>* a_Lights,
    SceneDataTableAccessor* a_SceneDataTable,
    unsigned a_InstanceId)
{

    const unsigned numTriangles = a_IndexBufferSize / 3; //Num of triangles we are processing (max bound).
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; //First triangle index this thread executes on.
    const unsigned int stride = blockDim.x * gridDim.x; //Stride is too handle the entire buffer of elements.
    //If there are more elements than the total amount of threads. 

    for(unsigned int triangleIndex = index; triangleIndex < numTriangles; triangleIndex += stride)
    {
        const unsigned baseIndex = triangleIndex * 3; //We run this function for each triangle, triangle has 3 vertices.

        assert(a_SceneDataTable->GetTableEntry<DevicePrimitiveInstance>(a_InstanceId) != nullptr);
        const auto devicePrimitiveInstance = a_SceneDataTable->GetTableEntry<DevicePrimitiveInstance>(a_InstanceId);
        const auto devicePrimitive = devicePrimitiveInstance->m_Primitive;
        const auto indices = devicePrimitive.m_IndexBuffer;
        const auto vertices = devicePrimitive.m_VertexBuffer;
        const auto emissiveMarks = devicePrimitive.m_EmissiveBuffer;

        //TODO this can be optimized in case of override.
        //check first vertex of triangle to see if its in emissive buffer
        if ((devicePrimitiveInstance->m_EmissionMode == Lumen::EmissionMode::ENABLED && emissiveMarks[triangleIndex] == true) || devicePrimitiveInstance->m_EmissionMode == Lumen::EmissionMode::OVERRIDE)
        {
        	
            const unsigned index0 = indices[baseIndex + 0];
            const unsigned index1 = indices[baseIndex + 1];
            const unsigned index2 = indices[baseIndex + 2];

            const Vertex& vert0 = vertices[index0];
            const Vertex& vert1 = vertices[index1];
            const Vertex& vert2 = vertices[index2];

            //check if normal is here
            float4 tempWorldPos;
            WaveFront::TriangleLight light{};
            //Need it this way because need float3 to float4 and back conversions
            //Light vertex 0
            tempWorldPos = make_float4(vert0.m_Position, 1.f);
            tempWorldPos = devicePrimitiveInstance->m_Transform * tempWorldPos;
            light.p0 = make_float3(tempWorldPos);
            //Light vertex 1
            tempWorldPos = { vert1.m_Position.x, vert1.m_Position.y, vert1.m_Position.z, 1.0f };
            tempWorldPos = devicePrimitiveInstance->m_Transform * tempWorldPos;
            light.p1 = make_float3(tempWorldPos);
            //Light vertex 2
            tempWorldPos = { vert2.m_Position.x, vert2.m_Position.y, vert2.m_Position.z, 1.0f };
            tempWorldPos = devicePrimitiveInstance->m_Transform * tempWorldPos;
            light.p2 = make_float3(tempWorldPos);

            constexpr float oneThird = 1.f / 3.f;
            const float2 UVCentroid = (vert0.m_UVCoord + vert1.m_UVCoord + vert2.m_UVCoord) * oneThird;

            auto mat = devicePrimitiveInstance->m_Primitive.m_Material;

            //float4 diffuseColor = a_Mat->m_DiffuseColor;

            //if (diffuseTexture)
            //{
            //    diffuseColor *= tex2D<float4>(diffuseTexture, UVCentroid.x, UVCentroid.y);
            //}

            //Emissive mode
            float4 emissive = make_float4(0.f);

            //When enabled, just read the GLTF data and scale it accordingly.
            if (devicePrimitiveInstance->m_EmissionMode == Lumen::EmissionMode::ENABLED)
            {
                emissive = tex2D<float4>(mat->m_EmissiveTexture, UVCentroid.x, UVCentroid.y);
                emissive *= mat->m_MaterialData.m_Emissive * devicePrimitiveInstance->m_EmissiveColorAndScale.w;
            }
            //When override, take the ovverride emissive color and scale it up.
            else if (devicePrimitiveInstance->m_EmissionMode == Lumen::EmissionMode::OVERRIDE)
            {
                emissive = devicePrimitiveInstance->m_EmissiveColorAndScale * devicePrimitiveInstance->m_EmissiveColorAndScale.w;
            }

        	//Only add actually emissive lights.
        	if(emissive.x > 0.f || emissive.y > 0.f || emissive.z > 0.f)
        	{
                light.radiance = make_float3(emissive);
                light.normal = (vert0.m_Normal + vert1.m_Normal + vert2.m_Normal) * oneThird;

        		//Transform the normal to world space.
                float4 tempNormal = make_float4(light.normal, 0.0f);   //No translation for normals so last components = 0.f.
                tempNormal = devicePrimitiveInstance->m_Transform * tempNormal;
                light.normal = normalize(make_float3(tempNormal));

                const float3 vec1 = light.p0 - light.p1;
                const float3 vec2 = light.p0 - light.p2;

                light.area = sqrtf(
                    pow((vec1.y * vec2.z - vec2.y * vec1.z), 2) +
                    pow((vec1.x * vec2.z - vec2.x * vec1.z), 2) +
                    pow((vec1.x * vec2.y - vec2.x * vec1.y), 2)
                ) / 2.0f;

                a_Lights->Add(&light);
        	}

            //Add light to lightbuffer, but to know where the end of the buffer is, keep track of lightIndex.
            //Dont know how many lights have already been added to buffer.
        }
        
    }

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