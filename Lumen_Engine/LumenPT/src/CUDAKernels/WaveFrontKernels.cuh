#pragma once
#include "WaveFrontKernels/CPUDataBufferKernels.cuh"
#include "WaveFrontKernels/CPUShadingKernels.cuh"

using namespace WaveFront;

//Helper functions.

GPU_ONLY INLINE unsigned int WangHash(unsigned int a_S);

GPU_ONLY INLINE unsigned int RandomInt(unsigned int& a_S);

GPU_ONLY INLINE float RandomFloat(unsigned int& a_S);