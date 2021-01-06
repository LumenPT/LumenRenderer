#include "Mesh.h"

#include "MemoryBuffer.h"

Mesh::Mesh(std::unique_ptr<MemoryBuffer> a_VertexBuffer, std::unique_ptr<MemoryBuffer> a_IndexBuffer, OptixTraversableHandle a_TraversableHandle)
    : m_VertBuffer(std::move(a_VertexBuffer))
    , m_IndexBuffer(std::move(a_IndexBuffer))
    , m_TraversableHandle(a_TraversableHandle)
{
    
}
