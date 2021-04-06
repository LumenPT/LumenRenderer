#pragma once
#include "Renderer/ILumenResources.h"
#include "ShaderBindingTableRecord.h"
#include "SceneDataTableEntry.h"
#include "AccelerationStructure.h"

#include "../../src/Shaders/CppCommon/ModelStructs.h"

#include "Optix/optix_types.h"

#include <memory>

class MemoryBuffer;

// Extension of base class that takes care of the implementation details of the APIs that are used for pathtracing
class PTPrimitive : public Lumen::ILumenPrimitive
{
public:
	PTPrimitive(std::unique_ptr<MemoryBuffer> a_VertexBuffer,
		std::unique_ptr<MemoryBuffer> a_IndexBuffer, 
		std::unique_ptr<MemoryBuffer> a_BoolBuffer,
		std::unique_ptr<AccelerationStructure> a_GeometryAccelerationStructure);

	~PTPrimitive();

	std::unique_ptr<MemoryBuffer> m_VertBuffer; // Memory buffer containing the vertex data of the primitive
	std::unique_ptr<MemoryBuffer> m_IndexBuffer; // Memory buffer containing the index data if such exists
	std::unique_ptr<MemoryBuffer> m_BoolBuffer; // ???????
	// The geometry acceleration structure of the primitive. This is used when creating the instance acceleration structure of the owner mesh
	std::unique_ptr<AccelerationStructure> m_GeometryAccelerationStructure; 

	// Handle to the primitive's entry in the scene data table
    // Controls what data is associated with the primitive and what instance ID it gets during IAS construction
	SceneDataTableEntry<DevicePrimitive> m_SceneDataTableEntry;
private:
};
