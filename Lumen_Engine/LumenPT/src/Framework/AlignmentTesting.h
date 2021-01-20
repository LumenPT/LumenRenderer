#pragma once
#include "MemoryBuffer.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include <device_launch_parameters.h>

#include "../../vendor/Include/Cuda/cuda/helpers.h"
#include "../CUDAKernels/AlignmentTesting.cuh"
#include "../Shaders/CppCommon/AlignmentTestStructs.h"

class AlignmentTest
{
public:
	AlignmentTest() = default;

	/*
	 * Test if alignment messes up with different structs.
	 */
    void RunTest()
	{
		constexpr int offset = 16;
		constexpr int numElements = 3;
		m_MemBuffer.Resize(3 * sizeof(Test1) + offset);

		//Fill data with expected values.
		Test1 data[numElements];
		data[0] = Test1{1, 2, 2, 2,  3, 3, 3, 3};
		data[1] = Test1{ 4, 5, 5, 5,  6, 6, 6, 6 };
		data[2] = Test1{7, 8, 9, 10,  11, 11, 11, 11 };

		//Upload to the GPU.
		m_MemBuffer.Write(data, 3 * sizeof(Test1), offset);
		char* ptr = (char*)m_MemBuffer.GetDevicePtr();
		ptr += offset;

		PrintTest1OnGPU(reinterpret_cast<Test1*>(ptr), numElements);
		cudaDeviceSynchronize();
	}

private:
	MemoryBuffer m_MemBuffer;
};
