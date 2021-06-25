#pragma once
#include "../CudaDefines.h"
#include <cinttypes>

#include <cuda_runtime_api.h>
#include <Cuda/cuda/helpers.h>

class GPULightDataBuffer
{

public:

    CPU_GPU
        GPULightDataBuffer() = default;

    CPU_ONLY
        GPULightDataBuffer(cudaSurfaceObject_t a_DataSurface)
        :
        m_IndexCounter(0),
        m_DataSurface(a_DataSurface)
    {}

    GPU_ONLY uint32_t AddEmissives(uint32_t a_NumEmissives)
    {
        uint32_t currentIndex = atomicAdd(&m_IndexCounter, a_NumEmissives);
    }


    cudaSurfaceObject_t m_DataSurface;

private:

    uint32_t m_IndexCounter;

};
