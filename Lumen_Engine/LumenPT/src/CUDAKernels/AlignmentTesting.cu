#include "AlignmentTesting.cuh"
#include "cuda_runtime_api.h"
#include "cuda.h"
#include "../Shaders/CppCommon/AlignmentTestStructs.h"
#include <cstdio>

__host__ void PrintTest1OnGPU(Test1* a_Ptr, int a_NumElements)
{
    PrintTest1OnGPUInternal<<<1, 1>>>(a_Ptr, a_NumElements);
}

__global__ void PrintTest1OnGPUInternal(Test1* a_Ptr, int a_NumElements)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < a_NumElements; i += stride)
    {
        Test1& test = a_Ptr[i];

        printf("Element index: %i\n", i);
        printf("    F1: %f\n", test.f1);
        printf("    F3: %f %f %f\n", test.f3.x, test.f3.y, test.f3.z);
        printf("    F42: %f %f %f %f\n", test.f42.x, test.f42.y, test.f42.z, test.f42.w);
    }
}
