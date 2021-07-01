#pragma once
#include "../../Shaders/CppCommon/CudaDefines.h"
#include "../../Shaders/CppCommon/WaveFrontDataStructs.h"
#include "../../Framework/LightDataBuffer.h"

using namespace WaveFront;

CPU_ONLY void FindEmissives(
    const Vertex* a_Vertices,
    const uint32_t* a_Indices,
    bool* a_Emissives,
    const DeviceMaterial* a_Mat,
    const uint32_t a_IndexBufferSize,
    unsigned int& a_NumLights
);

CPU_ONLY void AddToLightBuffer(
    const uint32_t a_IndexBufferSize,
    WaveFront::AtomicBuffer<WaveFront::TriangleLight>* a_Lights,
    SceneDataTableAccessor* a_SceneDataTable,
    unsigned a_InstanceId
);

CPU_ONLY void BuildLightDataBufferOnGPU(
    LightInstanceData* a_InstanceData,
    uint32_t a_NumInstances,
    uint32_t a_AverageNumTriangles,
    const SceneDataTableAccessor* a_SceneDataTable,
    //GPULightDataBuffer a_DataBuffer
    WaveFront::AtomicBuffer<WaveFront::TriangleLight>* a_DataBufferDevPtr
);