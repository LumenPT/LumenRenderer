#include "PTPrimitive.h"
#include "AccelerationStructure.h"
#include "MemoryBuffer.h"

PTPrimitive::PTPrimitive(std::unique_ptr<MemoryBuffer> a_VertexBuffer, std::unique_ptr<MemoryBuffer> a_IndexBuffer, std::unique_ptr<MemoryBuffer> a_BoolBuffer, std::unique_ptr<AccelerationStructure> a_GeometryAccelerationStructure)
    : m_VertBuffer(std::move(a_VertexBuffer))
    , m_IndexBuffer(std::move(a_IndexBuffer))
    , m_BoolBuffer(std::move(a_BoolBuffer))
    , m_GeometryAccelerationStructure(std::move(a_GeometryAccelerationStructure))
{
    
}

PTPrimitive::~PTPrimitive()
{
}
