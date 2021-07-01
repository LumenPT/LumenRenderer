#pragma once
#include "../../Shaders/CppCommon/CudaDefines.h"
#include "../../Shaders/CppCommon/WaveFrontDataStructs.h"
#include "../../Shaders/CppCommon/WaveFrontDataStructs/LightData.h"
#include "../../Shaders/CppCommon/WaveFrontDataStructs/AtomicBuffer.h"
#include <sutil/Matrix.h>

class SceneDataTableAccessor;
class PTMaterial;

CPU_ON_GPU void FindEmissivesGpu(
	const Vertex* a_Vertices, 
	const uint32_t* a_Indices,
	bool* a_Emissives,
	const DeviceMaterial* a_Mat,
	const uint32_t a_IndexBufferSize, 
	unsigned int* a_NumLights
);

CPU_ON_GPU void AddToLightBufferGpu(
	const uint32_t a_IndexBufferSize, 
	WaveFront::AtomicBuffer<WaveFront::TriangleLight>* a_Lights,
	SceneDataTableAccessor* a_SceneDataTable,
	unsigned a_InstanceId
);

//CPU_ON_GPU void AddToLightBuffer2();