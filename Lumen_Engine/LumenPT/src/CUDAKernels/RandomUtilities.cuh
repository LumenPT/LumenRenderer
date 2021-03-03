#pragma once

#include <cuda_runtime.h>

__host__ __device__ __inline__ unsigned int WangHash(unsigned int a_S)
{
    a_S = (a_S ^ 61) ^ (a_S >> 16), a_S *= 9, a_S = a_S ^ (a_S >> 4), a_S *= 0x27d4eb2d, a_S = a_S ^ (a_S >> 15); return a_S;
}

__host__ __device__ __inline__ unsigned int RandomInt(unsigned int& a_S)
{
    a_S ^= a_S << 13, a_S ^= a_S >> 17, a_S ^= a_S << 5; return a_S;
}

__host__ __device__ __inline__ float RandomFloat(unsigned int& a_S)
{
    return RandomInt(a_S) * 2.3283064365387e-10f;
}