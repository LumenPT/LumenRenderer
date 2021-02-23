#include <device_launch_parameters.h>

#include "../../vendor/Include/Cuda/cuda/helpers.h"

struct Test1;
__host__ void PrintTest1OnGPU(Test1* a_Ptr, int a_NumElements);

__global__ void PrintTest1OnGPUInternal(Test1* a_Ptr, int a_NumElements);