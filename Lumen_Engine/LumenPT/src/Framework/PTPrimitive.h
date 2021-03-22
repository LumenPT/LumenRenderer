#pragma once
#include "Renderer/ILumenResources.h"
#include "ShaderBindingTableRecord.h"
#include "SceneDataTableEntry.h"
#include "AccelerationStructure.h"

#include "../../src/Shaders/CppCommon/ModelStructs.h"

#include "Optix/optix_types.h"

#include <memory>

class MemoryBuffer;

class PTPrimitive : public Lumen::ILumenPrimitive
{
public:
	PTPrimitive(std::unique_ptr<MemoryBuffer> a_VertexBuffer,
		std::unique_ptr<MemoryBuffer> a_IndexBuffer, std::unique_ptr<AccelerationStructure> a_GeometryAccelerationStructure);

	~PTPrimitive();

	std::unique_ptr<MemoryBuffer> m_VertBuffer;
	std::unique_ptr<MemoryBuffer> m_IndexBuffer;
	std::unique_ptr<AccelerationStructure> m_GeometryAccelerationStructure;

	SceneDataTableEntry<DevicePrimitive> m_SceneDataTableEntry;
private:
};
