#pragma once

#include <cuda_runtime.h>

//Some defines to make the functions less scary and more readable

#define GPU_ONLY __device__ __forceinline__ //Runs on GPU only, available on GPU only.
#define CPU_GPU __global__ __forceinline__ //Runs on GPU, available on GPU and CPU.
#define CPU_ONLY __host__

