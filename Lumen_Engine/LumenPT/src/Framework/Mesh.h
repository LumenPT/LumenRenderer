#pragma once
#include "Lumen/Renderer/ILumenMesh.h"
#include "../../vendor/Include/Optix/optix_types.h"

#include <memory>


class VertexBuffer;

class Mesh : public ILumenMesh
{
	std::unique_ptr<VertexBuffer> m_VertBuffer;
	std::unique_ptr<VertexBuffer> m_IndexBuffer;
	OptixTraversableHandle m_TraversableHandle;
};
