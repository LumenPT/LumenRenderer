#include "GPUDataBufferKernels.cuh"
#include "../../Framework/LightDataBuffer.h"
#include "../../../../Lumen/src/Lumen/ModelLoading/MeshInstance.h"
#include "../../Shaders/CppCommon/WaveFrontDataStructs/LightData.h"
#include "../../Shaders/CppCommon/SceneDataTableAccessor.h"
#include <device_launch_parameters.h>

//Gets called numBlocks.x * numThreads.x * numBlocks.y * numThreads.y
CPU_ON_GPU
void BuildLightDataBufferGPU(
    LightInstanceData* a_InstanceData, 
    uint32_t a_NumInstances, 
    const SceneDataTableAccessor* a_SceneDataTable, 
    //GPULightDataBuffer a_DataBuffer)
    WaveFront::AtomicBuffer<WaveFront::TriangleLight>* a_DataBuffer)
{

    //X-axis handles the number of instances.
    const unsigned instanceIndex = blockIdx.x * blockDim.x + threadIdx.x;
    //Y-axis handles the triangles of each instance, so that there is not just one thread looping over all the triangles in an instance.
    const unsigned instanceThreadIndex = blockIdx.y * blockDim.y + threadIdx.y;
    //The number of threads to handle the triangles of an instance.
    const unsigned numThreadsPerInstance = gridDim.y * blockDim.y;

    //Dont try to build more instances than there are, which could happen if there is more threads than instances due to threads launching in blocks.
    if(instanceIndex < a_NumInstances) 
    {

        const LightInstanceData& instanceData = a_InstanceData[instanceIndex];
        
        //Y-axis handles the number of triangles per instance, how many triangles per thead for this instance?
        const unsigned numTrianglesPerThread = 
            static_cast<unsigned>(ceilf(static_cast<float>(instanceData.m_NumTriangles) /static_cast<float>(numThreadsPerInstance)));

        const unsigned startTriangleIndex = instanceThreadIndex * numTrianglesPerThread;

        if(startTriangleIndex < (instanceData.m_NumTriangles -1))
        {
            //Make sure that the numTriangles being processed does not cause to go out of range.
            const unsigned numTriangles =
                //If the start index + num of processed triangles per thread is smaller than total triangles.
                ((startTriangleIndex + numTrianglesPerThread) < instanceData.m_NumTriangles) ?
                //Take the number of processed triangles per thread.
                numTrianglesPerThread :
                //Otherwise, calculate the number of triangles to process for this thread.
                (instanceData.m_NumTriangles) - startTriangleIndex;

            //Get the current start index and reserve the number of triangles that will be processed.
            const uint32_t startIndexData = a_DataBuffer->ReserveIndices(numTriangles);
            
            BuildLightDataInstance(
                instanceData, 
                a_SceneDataTable,
                startTriangleIndex,
                numTriangles,
                startIndexData, 
                a_DataBuffer);

        }

    }

}


GPU_ONLY
void BuildLightDataInstance(
    const LightInstanceData& a_InstanceData,
    const SceneDataTableAccessor* a_SceneDataTable,
    uint32_t a_startTriangleIndex,
    uint32_t a_NumTriangles,
    uint32_t a_DataStartIndex,
    //cudaSurfaceObject_t a_DataBuffer)
    WaveFront::AtomicBuffer<WaveFront::TriangleLight>* a_DataBuffer)
{

    const auto primitiveInstance = a_SceneDataTable->GetTableEntry<DevicePrimitiveInstance>(a_InstanceData.m_DataTableIndex);
    const auto primitive = primitiveInstance->m_Primitive;
    const auto indices = primitive.m_IndexBuffer;
    const auto vertices = primitive.m_VertexBuffer;
    const auto emissives = primitive.m_EmissiveBuffer;
    assert(primitiveInstance != nullptr);

    for (unsigned int triangleIndex = 0; triangleIndex < a_NumTriangles; ++triangleIndex)
    {
        const unsigned baseTriangleIndex = a_startTriangleIndex + triangleIndex;
        const unsigned baseIndex = baseTriangleIndex * 3; //We run this function for each triangle, triangle has 3 vertices.

        //TODO this can be optimized in case of override.
        //check first vertex of triangle to see if its in emissive buffer
        if ((primitiveInstance->m_EmissionMode == Lumen::EmissionMode::ENABLED && emissives[baseTriangleIndex] == true) ||
            primitiveInstance->m_EmissionMode == Lumen::EmissionMode::OVERRIDE)
        {

            const unsigned index0 = indices[baseIndex + 0];
            const unsigned index1 = indices[baseIndex + 1];
            const unsigned index2 = indices[baseIndex + 2];

            const Vertex& vertex0 = vertices[index0];
            const Vertex& vertex1 = vertices[index1];
            const Vertex& vertex2 = vertices[index2];

            //check if normal is here
            float4 tempWorldPos;
            WaveFront::TriangleLightUint4_4 light{};
            //Need it this way because need float3 to float4 and back conversions
            //Light vertex 0
            tempWorldPos = make_float4(vertex0.m_Position, 1.f);
            tempWorldPos = primitiveInstance->m_Transform * tempWorldPos;
            light.m_TriangleLight.p0 = make_float3(tempWorldPos);
            //Light vertex 1
            tempWorldPos = make_float4(vertex1.m_Position, 1.f);
            tempWorldPos = primitiveInstance->m_Transform * tempWorldPos;
            light.m_TriangleLight.p1 = make_float3(tempWorldPos);
            //Light vertex 2
            tempWorldPos = make_float4(vertex2.m_Position, 1.f);
            tempWorldPos = primitiveInstance->m_Transform * tempWorldPos;
            light.m_TriangleLight.p2 = make_float3(tempWorldPos);

            constexpr float oneThird = 1.f / 3.f;
            const float2 UVCentroid = (vertex0.m_UVCoord + vertex1.m_UVCoord + vertex2.m_UVCoord) * oneThird;

            auto mat = primitiveInstance->m_Primitive.m_Material;

            //float4 diffuseColor = a_Mat->m_DiffuseColor;

            //if (diffuseTexture)
            //{
            //    diffuseColor *= tex2D<float4>(diffuseTexture, UVCentroid.x, UVCentroid.y);
            //}

            //Emissive mode
            float4 emissive = make_float4(0.f);

            //When enabled, just read the GLTF data and scale it accordingly.
            if (primitiveInstance->m_EmissionMode == Lumen::EmissionMode::ENABLED)
            {
                emissive = tex2D<float4>(mat->m_EmissiveTexture, UVCentroid.x, UVCentroid.y);
                emissive *= mat->m_MaterialData.m_Emissive * primitiveInstance->m_EmissiveColorAndScale.w;
            }
            //When override, take the ovverride emissive color and scale it up.
            else if (primitiveInstance->m_EmissionMode == Lumen::EmissionMode::OVERRIDE)
            {
                emissive = primitiveInstance->m_EmissiveColorAndScale * primitiveInstance->m_EmissiveColorAndScale.w;
            }

            //Only add actually emissive lights.
            if (emissive.x > 0.f || emissive.y > 0.f || emissive.z > 0.f)
            {
                light.m_TriangleLight.radiance = make_float3(emissive);
                light.m_TriangleLight.normal = (vertex0.m_Normal + vertex1.m_Normal + vertex2.m_Normal) * oneThird;

                //Transform the normal to world space.
                float4 tempNormal = make_float4(light.m_TriangleLight.normal, 0.0f);   //No translation for normals so last components = 0.f.
                tempNormal = primitiveInstance->m_Transform * tempNormal;
                light.m_TriangleLight.normal = normalize(make_float3(tempNormal));

                const float3 vec1 = light.m_TriangleLight.p0 - light.m_TriangleLight.p1;
                const float3 vec2 = light.m_TriangleLight.p0 - light.m_TriangleLight.p2;

                light.m_TriangleLight.area = sqrtf(
                    pow((vec1.y * vec2.z - vec2.y * vec1.z), 2) +
                    pow((vec1.x * vec2.z - vec2.x * vec1.z), 2) +
                    pow((vec1.x * vec2.y - vec2.x * vec1.y), 2)
                ) / 2.0f;

//#pragma unroll
//                for(unsigned layerIndex = 0; layerIndex < _countof(WaveFront::TriangleLightUint4_4::m_Uint4); ++layerIndex)
//                {
//                    surf1DLayeredwrite<uint4>(
//                        light.m_Uint4[layerIndex],
//                        a_DataBuffer,
//                        (a_DataStartIndex + triangleIndex) * sizeof(uint4),
//                        layerIndex,
//                        cudaBoundaryModeTrap);
//                    
//                }

                a_DataBuffer->Set((a_DataStartIndex + triangleIndex), light.m_TriangleLight);

            }
        }

    }

}

