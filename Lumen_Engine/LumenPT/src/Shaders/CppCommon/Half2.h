#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

union half2Ushort2
{
    half2 m_Half2;
    ushort2 m_Ushort2;

    __device__ __host__
    float2 AsFloat2() const
    {
        return __half22float2(m_Half2);
    }

};