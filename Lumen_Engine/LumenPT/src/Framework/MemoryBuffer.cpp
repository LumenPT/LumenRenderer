#include "MemoryBuffer.h"

#include "Cuda/cuda.h"
#include "Cuda/cuda_runtime.h"

#include <cassert>

MemoryBuffer::MemoryBuffer(size_t a_Size)
: m_Size(a_Size)
{
    cudaMalloc(reinterpret_cast<void**>(m_CudaPtr), m_Size);
};
MemoryBuffer::~MemoryBuffer()
{
    cudaFree(&m_CudaPtr);
};

CUdeviceptr& MemoryBuffer::operator*() { return m_CudaPtr; }

void MemoryBuffer::Write(const void* a_Data, size_t a_Size, size_t a_Offset)
{
    assert(a_Size + a_Offset <= m_Size);
    cudaMemcpy(reinterpret_cast<void*>(m_CudaPtr + a_Offset), a_Data, a_Size, cudaMemcpyKind::cudaMemcpyHostToDevice);
};

void MemoryBuffer::Read(void* a_Dst, size_t a_ReadSize, size_t a_SrcOffset) const
{
    assert(a_SrcOffset + a_ReadSize < m_Size);
    cudaMemcpy(a_Dst, reinterpret_cast<void*>(m_CudaPtr + a_SrcOffset), a_ReadSize, cudaMemcpyDeviceToHost);
};

void MemoryBuffer::CopyFrom(MemoryBuffer a_MemoryBuffer, size_t a_Size, size_t a_DstOffset, size_t a_SrcOffset)
{
    assert(a_DstOffset + a_Size < m_Size);
    assert(a_SrcOffset + a_Size < a_MemoryBuffer.GetSize());

    cudaMemcpy(reinterpret_cast<void*>(m_CudaPtr + a_DstOffset), reinterpret_cast<void*>(*a_MemoryBuffer + a_SrcOffset),
        a_Size, cudaMemcpyDeviceToDevice);
};

size_t MemoryBuffer::GetSize() const
{
    return m_Size;
}