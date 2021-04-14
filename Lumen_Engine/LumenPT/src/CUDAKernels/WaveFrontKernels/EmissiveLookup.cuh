#pragma once
#include "../../Shaders/CppCommon/CudaDefines.h"
#include "../../Shaders/CppCommon/WaveFrontDataStructs.h"

class Material;

CPU_ON_GPU void FindEmissives(const Vertex* a_Vertices, bool* a_EmissiveBools, const uint32_t* a_Indices, const DeviceMaterial* a_Mat, const uint8_t a_VertexBufferSize);