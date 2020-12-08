#include "MemoryBuffer.h"

#include "Cuda/cuda.h"
#include "Cuda/cuda_runtime.h"

#include <cassert>
#include <cstdint>
#include <cstdio>

void CudaCheck(cudaError_t err)
{
    if (err != 0)
    {
        printf("Cuda error %u", err);
    }
}

MemoryBuffer::MemoryBuffer(size_t a_Size)
: m_Size(a_Size)
{
    CudaCheck(cudaMalloc(&m_DevPtr, m_Size));

    //CudaCheck(cudaFree(m_DevPtr));

    m_CudaPtr = reinterpret_cast<CUdeviceptr>(m_DevPtr);
};

MemoryBuffer::~MemoryBuffer()
{
    CudaCheck(cudaFree(m_DevPtr));
};

CUdeviceptr& MemoryBuffer::operator*() { return m_CudaPtr; }

void MemoryBuffer::Write(const void* a_Data, size_t a_Size, size_t a_Offset)
{
    assert(a_Size + a_Offset <= m_Size);
    CudaCheck(cudaMemcpy(reinterpret_cast<void*>(reinterpret_cast<CUdeviceptr>(m_DevPtr) + a_Offset), a_Data, a_Size, cudaMemcpyKind::cudaMemcpyHostToDevice));
};

void MemoryBuffer::Read(void* a_Dst, size_t a_ReadSize, size_t a_SrcOffset) const
{
    assert(a_SrcOffset + a_ReadSize < m_Size);
    CudaCheck(cudaMemcpy(a_Dst, reinterpret_cast<void*>(reinterpret_cast<CUdeviceptr>(m_DevPtr) + a_SrcOffset), a_ReadSize, cudaMemcpyDeviceToHost));
};

void MemoryBuffer::CopyFrom(MemoryBuffer a_MemoryBuffer, size_t a_Size, size_t a_DstOffset, size_t a_SrcOffset)
{
    assert(a_DstOffset + a_Size < m_Size);
    assert(a_SrcOffset + a_Size < a_MemoryBuffer.GetSize());

    CudaCheck(cudaMemcpy(reinterpret_cast<void*>(reinterpret_cast<CUdeviceptr>(m_DevPtr) + a_DstOffset), reinterpret_cast<void*>(*a_MemoryBuffer + a_SrcOffset),
        a_Size, cudaMemcpyDeviceToDevice));
};

size_t MemoryBuffer::GetSize() const
{
    return m_Size;
}