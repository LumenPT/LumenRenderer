#pragma once
#include "../CudaDefines.h"
#include <Cuda/cuda/helpers.h>
#include <cassert>
#include "cuda_runtime_api.h"
#include "cuda/device_atomic_functions.h"

namespace WaveFront
{
/*
 * Buffer that allows atomic additions on a buffer.
 */
template<typename T>
struct AtomicBuffer
{
    //Functions
public:
    /*
     * Append data to the buffer.
     */
    GPU_ONLY void Add(T* a_Data)
    {
        //Add at index - 1 because the counter gives the total size, which starts at 1. 
        const unsigned index = atomicAdd(counter, 1);
        data[index - 1] = *a_Data;
    }

    /*
     * Set data in the buffer, bypassing the atomic operation.
     */
    GPU_ONLY void Set(int a_Index, T* a_Data)
    {
        data[a_Index] = a_Data;
    }

    /*
     * Set the value of the counter.
     * This bypasses atomic operations and should only be used when data is added by setting at specific indices.
     */
    GPU_ONLY void SetCounter(int a_Value)
    {
        counter = a_Value;
    }

    /*
     * Reset the counter back to 0.
     */
    GPU_ONLY void Reset()
    {
        counter = 0;
    }

    GPU_ONLY unsigned GetSize() const
    {
        return counter;
    }

    GPU_ONLY T* GetData(int a_Index)
    {
        assert(a_Index <= counter);
        return &data[a_Index];
    }

    //Data
public:
    unsigned counter;
    T data[];
};

}