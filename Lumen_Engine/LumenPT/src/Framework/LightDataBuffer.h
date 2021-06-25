#pragma once
//#include "GPUTexture.h"
#include "MemoryBuffer.h"
#include "../Shaders/CppCommon/WaveFrontDataStructs/LightData.h"
#include "../Shaders/CppCommon/WaveFrontDataStructs/AtomicBuffer.h"
#include "../Shaders/CppCommon/CudaDefines.h"
#include <memory>

class SceneDataTableAccessor;
class PTScene;

struct LightInstanceData
{
    uint32_t m_DataTableIndex;
    uint32_t m_NumTriangles;
    uint32_t m_NumEmissives;
};

class LightDataBuffer
{

public:

    LightDataBuffer(uint32_t a_BufferSize);


    unsigned int BuildLightDataBuffer(
        const std::shared_ptr<const PTScene>& a_Scene,
        const SceneDataTableAccessor* a_SceneDataTableAccessor);

    const MemoryBuffer* GetDataBuffer() const;

private:

    uint32_t m_Size;

    //GPUTexture<float4> m_DataBuffer;
    MemoryBuffer m_DataBuffer;

};