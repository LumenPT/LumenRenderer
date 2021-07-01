#include "LightDataBuffer.h"
#include "SceneDataTable.h"
#include "PTScene.h"
#include "PTMeshInstance.h"
#include "PTPrimitive.h"
#include "../CUDAKernels/WaveFrontKernels/CPUDataBufferKernels.cuh"


LightDataBuffer::LightDataBuffer(uint32_t a_BufferSize)
    :
m_Size(a_BufferSize),
//m_DataBuffer(
//    cudaExtent{a_BufferSize, 0, _countof(TriangleLightUint4_4::m_Uint4)},
//    cudaArrayLayered | cudaArraySurfaceLoadStore,
//    cudaTextureDesc{
//        {cudaAddressModeClamp, cudaAddressModeClamp, cudaAddressModeClamp},
//        cudaFilterModePoint,
//        cudaReadModeElementType,
//        0,
//        {0.f, 0.f, 0.f, 0.f},
//        0,
//        0,
//        cudaFilterModePoint,
//        0.f,
//        0.f,
//        0.f,
//        0},
//    nullptr)
m_DataBuffer()
{

    CreateAtomicBuffer<TriangleLight>(&m_DataBuffer, m_Size);

}


unsigned int LightDataBuffer::BuildLightDataBuffer(
    const std::shared_ptr<const PTScene>& a_Scene,
    const SceneDataTableAccessor* a_SceneDataTableAccessor)
{

    ResetAtomicBuffer<TriangleLight>(&m_DataBuffer);

    std::vector<LightInstanceData> lightInstanceData{};
    unsigned int numEmissivePrimitives = 0;
    unsigned int totalNumEmissive = 0;
    float averageTrianglesPerPrim = 0;

    for (const std::unique_ptr<Lumen::MeshInstance>& meshInstance : a_Scene->m_MeshInstances)
    {
        //Only run when emission is not disabled, and override is active OR the GLTF has specified valid emissive triangles and mode is set to ENABLED.
        if (meshInstance->GetEmissionMode() != Lumen::EmissionMode::DISABLED &&
            ((meshInstance->GetEmissionMode() == Lumen::EmissionMode::ENABLED && meshInstance->GetMesh()->GetEmissiveness()) ||
                meshInstance->GetEmissionMode() == Lumen::EmissionMode::OVERRIDE))
        {

            PTMeshInstance* ptMeshInstance = reinterpret_cast<PTMeshInstance*>(meshInstance.get());

            //Loop over all instances.

            for (auto& prim : ptMeshInstance->GetMesh()->m_Primitives)
            {
                const PTPrimitive* ptPrim = reinterpret_cast<PTPrimitive*>(prim.get());

                if (ptPrim->m_ContainEmissive || ptMeshInstance->GetEmissionMode() == Lumen::EmissionMode::OVERRIDE)
                {

                    //Find the primitive instance in the data table.
                    const auto& entryMap = ptMeshInstance->GetInstanceEntryMap();
                    const auto entry = &entryMap.at(prim.get());

                    const uint32_t dataTableIndex = entry->m_TableIndex;
                    const uint32_t numIndices = ptPrim->m_IndexBuffer->GetSize() / sizeof(uint32_t);
                    const uint32_t numTriangles = numIndices / 3; //Should always be a whole number as we only have triangle meshes (3 indices per triangle in the index buffer).
                    const uint32_t numEmissives = ptPrim->m_NumLights;

                    averageTrianglesPerPrim = ((averageTrianglesPerPrim * static_cast<float>(numEmissivePrimitives)) +
                        static_cast<float>(numTriangles)) /
                        static_cast<float>(numEmissivePrimitives + 1);
                    numEmissivePrimitives++;
                    totalNumEmissive += numEmissives;

                    lightInstanceData.emplace_back(LightInstanceData{ dataTableIndex, numTriangles, numEmissives });

                }
                else continue;

            }

        }
        else continue;
    }

    if (totalNumEmissive > m_Size) //Too many emissives to be stored in the buffer. Trim down the instances that get put into the buffer until it fits.
    {
        std::vector<LightInstanceData>::const_reverse_iterator iterator = lightInstanceData.crbegin();

        for (; iterator < lightInstanceData.crend(); ++iterator)
        {
            totalNumEmissive -= iterator->m_NumEmissives;
            if (totalNumEmissive < m_Size)
            {
                break;
            }
        }

        //first increase reverse-iterator as it physically points to the element after the one it logically points to.
        // https://stackoverflow.com/questions/14760134/why-does-removing-the-first-element-of-a-list-invalidate-rend/14760316#14760316
        lightInstanceData.erase((++iterator).base(), lightInstanceData.end());

    }

    const MemoryBuffer gpuInstanceData(lightInstanceData);

    BuildLightDataBufferOnGPU(
        gpuInstanceData.GetDevicePtr<LightInstanceData>(),
        lightInstanceData.size(),
        static_cast<uint32_t>(std::roundf(averageTrianglesPerPrim)),
        a_SceneDataTableAccessor,
        //GPULightDataBuffer(m_DataBuffer.GetSurfaceObject()));
        m_DataBuffer.GetDevicePtr<AtomicBuffer<TriangleLight>>());

    return totalNumEmissive;

}

const MemoryBuffer* LightDataBuffer::GetDataBuffer() const
{

    return &m_DataBuffer;

}
