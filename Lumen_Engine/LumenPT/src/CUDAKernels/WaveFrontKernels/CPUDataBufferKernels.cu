#include "CPUDataBufferKernels.cuh"
#include "GPUDataBufferKernels.cuh"
#include "GPUEmissiveLookup.cuh"
#include <cmath>

CPU_ONLY void FindEmissives(
    const Vertex* a_Vertices,
    const uint32_t* a_Indices,
    bool* a_Emissives,
    const DeviceMaterial* a_Mat,
    const uint32_t a_IndexBufferSize,
    unsigned int& a_NumLights)
{
    unsigned int* numLightsPtr;
    cudaMalloc(&numLightsPtr, sizeof(unsigned int));

    FindEmissivesGpu << <1, 1 >> > (a_Vertices, a_Indices, a_Emissives, a_Mat, a_IndexBufferSize, numLightsPtr);
    cudaDeviceSynchronize();
    cudaMemcpy(&a_NumLights, numLightsPtr, sizeof(unsigned int), cudaMemcpyDeviceToHost);
}

CPU_ONLY void AddToLightBuffer(
    const uint32_t a_IndexBufferSize,
    WaveFront::AtomicBuffer<WaveFront::TriangleLight>* a_Lights,
    SceneDataTableAccessor* a_SceneDataTable,
    unsigned a_InstanceId)
{
    const unsigned int numTriangles = a_IndexBufferSize / 3;
    const int blockSize = 256;
    const int numBlocks = (numTriangles + blockSize - 1) / blockSize;

    AddToLightBufferGpu<< <numBlocks, blockSize >> > (a_IndexBufferSize, a_Lights, a_SceneDataTable, a_InstanceId);
}

CPU_ONLY void BuildLightDataBufferOnGPU(
    LightInstanceData* a_InstanceData, 
    uint32_t a_NumInstances,
    uint32_t a_AverageNumTriangles,
    const SceneDataTableAccessor* a_SceneDataTable, 
    //GPULightDataBuffer a_DataBuffer)
    WaveFront::AtomicBuffer<WaveFront::TriangleLight>* a_DataBufferDevPtr)
{

    //Block dimension is 2d.
    //The x-axis handles the number of instances.
    //The y-axis handles the number of triangles.
    const dim3 blockSize = {8, 64, 1};

    const unsigned gridWidth = static_cast<unsigned>(std::ceil(static_cast<float>(a_NumInstances) / static_cast<float>(blockSize.x)));
    const unsigned gridHeight = static_cast<unsigned>(std::ceil(static_cast<float>(a_AverageNumTriangles) / static_cast<float>(blockSize.y)));

    const dim3 gridSize = { gridWidth, gridHeight, 1 };

    BuildLightDataBufferGPU<<<gridSize, blockSize>>>(a_InstanceData, a_NumInstances, a_SceneDataTable, a_DataBufferDevPtr);

}
