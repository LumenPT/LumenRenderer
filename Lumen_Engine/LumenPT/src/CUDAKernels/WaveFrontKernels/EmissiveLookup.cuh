#pragma once
#include "../../Shaders/CppCommon/CudaDefines.h"
#include "../../Shaders/CppCommon/WaveFrontDataStructs.h"
#include "../../Shaders/CppCommon/WaveFrontDataStructs/LightDataBuffer.h"
#include "../../Shaders/CppCommon/WaveFrontDataStructs/AtomicBuffer.h"

class PTMaterial;

CPU_ON_GPU void FindEmissives(const Vertex* a_Vertices, bool* a_EmissiveBools, const uint32_t* a_Indices, const DeviceMaterial* a_Mat, const uint8_t a_VertexBufferSize, unsigned int& a_NumLights);

CPU_ON_GPU void AddToLightBuffer(const Vertex* a_Vertices, const uint32_t* a_Indices, const bool* a_Emissives, const uint8_t a_VertexBufferSize, WaveFront::AtomicBuffer<WaveFront::TriangleLight>* a_Lights, /*unsigned int& a_LightIndex,*/ sutil::Matrix4x4 a_TransformMat);