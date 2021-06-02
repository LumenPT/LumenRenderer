#pragma once
//#include <device_launch_parameters.h>
//
//#include "../../vendor/Include/Cuda/cuda/helpers.h"
//#include "../Shaders/CppCommon/WaveFrontDataStructs/GPUDataBuffers.h"
//
///*
// * The data structure that handles storing and retrieving instance data on the GPU.
// */
//class GPUSceneData
//{
//public:
//	/*
//	 * Get the instance for the given ID.
//	 * Returns nullptr when not found.
//	 */
//	__device__ DeviceInstanceData* GetInstance(unsigned int a_InstanceId);
//
//	/*
//	 * Create a new instance for the given data.
//	 * Returns the unique ID for the data.
//	 */
//	__device__ unsigned int CreateInstance(DeviceInstanceData* a_Data);
//
//	/*
//	 * Delete the instance data with the given ID.
//	 */
//	__device__ void DeleteInstance(unsigned int a_InstanceId);
//
//private:
//	//TODO add counter and instance adding etc.
//	DeviceInstanceData m_Data[];
//
//};
//
//__host__ void CreateGPUSceneData(MemoryBuffer& a_Buffer, size_t a_Size)
//{
//	//TODO allocate enough memory. Maybe dynamically grow? Some sort of unique ptr implementation?
//	a_Buffer.Resize(a_Size * sizeof(DeviceInstanceData) + sizeof(GPUSceneData));
//}