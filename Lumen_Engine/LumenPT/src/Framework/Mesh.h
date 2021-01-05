#pragma once
#include "Renderer/ILumenMesh.h"
#include "Optix/optix_types.h"

#include <memory>

class Mesh : public ILumenMesh
{
	Mesh(std::unique_ptr<MemoryBuffer> a_VertexBuffer,
		std::unique_ptr<MemoryBuffer> a_IndexBuffer, OptixTraversableHandle a_TraversableHandle);

	std::unique_ptr<MemoryBuffer> m_VertBuffer;
	std::unique_ptr<MemoryBuffer> m_IndexBuffer;
	OptixTraversableHandle m_TraversableHandle;
};
