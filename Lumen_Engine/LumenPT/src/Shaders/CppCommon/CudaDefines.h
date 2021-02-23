#pragma once
#include <cuda_runtime.h>

//Some defines to make the functions less scary and more readable

#define GPU_ONLY __device__         //Runs on GPU only, available on GPU only.
#define CPU_ONLY __host__			//Runs on the CPU only.
#define CPU_GPU __host__ __device__	//Available on GPU and CPU. runs on calling device.
#define CPU_ON_GPU __global__ 	    //Runs on GPU, available on and CPU.
#define INLINE __forceinline__