#pragma once
#include "../../Shaders/CppCommon/CudaDefines.h"
#include "../../Shaders/CppCommon/WaveFrontDataStructs.h"
#include "../../Framework/LightDataBuffer.h"

class GPULightDataBuffer;

CPU_ON_GPU
void BuildLightDataBufferGPU(
    LightInstanceData* a_InstanceData,
    uint32_t a_NumInstances,
    const SceneDataTableAccessor* a_SceneDataTable,
    //GPULightDataBuffer a_DataBuffer);
    WaveFront::AtomicBuffer<WaveFront::TriangleLight>* a_DataBuffer);

GPU_ONLY
void BuildLightDataInstance(
    const LightInstanceData& a_InstanceData,
    const SceneDataTableAccessor* a_SceneDataTable,
    uint32_t a_startTriangleIndex,
    uint32_t a_NumTriangles,
    uint32_t a_DataStartIndex,
    //cudaSurfaceObject_t a_DataBuffer);
    WaveFront::AtomicBuffer<WaveFront::TriangleLight>* a_DataBuffer);
