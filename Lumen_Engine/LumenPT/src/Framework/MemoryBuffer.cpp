#include "MemoryBuffer.h"

#include "Cuda/cuda.h"
#include "Cuda/cuda_runtime.h"

#include <cassert>
#include <cstdint>
#include <cstdio>

void CudaCheck(cudaError_t err)
{
#ifdef _DEBUG
    if (err != 0)
    {
        printf("Cuda error %u", err);
        abort();
    }
#else
    err;
#endif
}

MemoryBuffer::MemoryBuffer(size_t a_Size)
    : m_Size(a_Size)
    , m_DevPtr(nullptr)
{  
    Resize(a_Size);
};

MemoryBuffer::~MemoryBuffer()
{
    CudaCheck(cudaFree(m_DevPtr));
}

MemoryBuffer::MemoryBuffer(MemoryBuffer&& a_Other)
{
    m_CudaPtr = a_Other.m_CudaPtr;
    m_Size = a_Other.m_Size;

    a_Other.m_CudaPtr = 0;
};

CUdeviceptr& MemoryBuffer::operator*() { return m_CudaPtr; }

void MemoryBuffer::Write(const void* a_Data, size_t a_Size, size_t a_Offset)
{
    assert(a_Size + a_Offset <= m_Size);
    CudaCheck(cudaMemcpy(reinterpret_cast<void*>(reinterpret_cast<CUdeviceptr>(m_DevPtr) + a_Offset), a_Data, a_Size, cudaMemcpyKind::cudaMemcpyHostToDevice));
};

void MemoryBuffer::Read(void* a_Dst, size_t a_ReadSize, size_t a_SrcOffset) const
{
    assert(a_SrcOffset + a_ReadSize <= m_Size);
    CudaCheck(cudaMemcpy(a_Dst, reinterpret_cast<void*>(reinterpret_cast<CUdeviceptr>(m_DevPtr) + a_SrcOffset), a_ReadSize, cudaMemcpyDeviceToHost));
};

void MemoryBuffer::CopyFrom(MemoryBuffer& a_MemoryBuffer, size_t a_Size, size_t a_DstOffset, size_t a_SrcOffset)
{
    assert(a_DstOffset + a_Size < m_Size);
    assert(a_SrcOffset + a_Size < a_MemoryBuffer.GetSize());

    CudaCheck(cudaMemcpy(reinterpret_cast<void*>(reinterpret_cast<CUdeviceptr>(m_DevPtr) + a_DstOffset), reinterpret_cast<void*>(*a_MemoryBuffer + a_SrcOffset),
        a_Size, cudaMemcpyDeviceToDevice));
}

void MemoryBuffer::Resize(size_t a_NewSize)
{
    if (m_DevPtr)
        cudaFree(m_DevPtr);

    cudaMalloc(&m_DevPtr, a_NewSize);

    m_Size = a_NewSize;
};

size_t MemoryBuffer::GetSize() const
{
    return m_Size;
}

unsigned long long MemoryBuffer::GetCUDAPtr() const
{

    return m_CudaPtr;

}
