#pragma once
#include "../../Shaders/CppCommon/CudaDefines.h"
#include "../../Shaders/CppCommon/WaveFrontDataStructs.h"
#include "../../Shaders/CppCommon/WaveFrontDataStructs/LightData.h"
#include "../../Shaders/CppCommon/WaveFrontDataStructs/AtomicBuffer.h"
#include <sutil/Matrix.h>

class SceneDataTableAccessor;
class PTMaterial;

CPU_ONLY void FindEmissivesWrap(
	const Vertex* a_Vertices,
	const uint32_t* a_Indices,
	bool* a_Emissives,
	const DeviceMaterial* a_Mat,
	const uint32_t a_IndexBufferSize,
	unsigned int& a_NumLights
);

CPU_ON_GPU void FindEmissives(
	const Vertex* a_Vertices, 
	const uint32_t* a_Indices,
	bool* a_Emissives,
	const DeviceMaterial* a_Mat,
	const uint32_t a_IndexBufferSize, 
	unsigned int* a_NumLights
);

CPU_ONLY void AddToLightBufferWrap(
	const Vertex* a_Vertices,
	const uint32_t* a_Indices,
	const bool* a_Emissives,
	const uint32_t a_IndexBufferSize,
	WaveFront::AtomicBuffer<WaveFront::TriangleLight>* a_Lights,
	SceneDataTableAccessor* a_SceneDataTable,
	unsigned a_InstanceId
);

CPU_ON_GPU void AddToLightBuffer(
	const Vertex* a_Vertices,
	const uint32_t* a_Indices,
	const bool* a_Emissives,
	const uint32_t a_IndexBufferSize, 
	WaveFront::AtomicBuffer<WaveFront::TriangleLight>* a_Lights,
	SceneDataTableAccessor* a_SceneDataTable,
	unsigned a_InstanceId
);

//CPU_ON_GPU void AddToLightBuffer2();