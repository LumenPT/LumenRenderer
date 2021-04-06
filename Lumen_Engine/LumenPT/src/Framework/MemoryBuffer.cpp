#include "MemoryBuffer.h"

#include "Cuda/cuda.h"
#include "Cuda/cuda_runtime.h"

#include <cassert>
#include <cstdint>
#include <cstdio>

// The memory buffer is usually the first class to pick up on a nuked CUDA context after something goes wrong
// This is because it makes extensive use of the CUDA toolkit.
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
    // Simply free the previously allocated GPU memory
    CudaCheck(cudaFree(m_DevPtr));
}

MemoryBuffer::MemoryBuffer(MemoryBuffer&& a_Other)
{
    m_CudaPtr = a_Other.m_CudaPtr;
    m_Size = a_Other.m_Size;

    a_Other.m_CudaPtr = 0;
};

// CUdeviceptr is typedef of unsigned long long
CUdeviceptr& MemoryBuffer::operator*() { return m_CudaPtr; }

void MemoryBuffer::Write(const void* a_Data, size_t a_Size, size_t a_Offset)
{
    // Assert that the offset won't cause us to go outside of the allocated memory
    assert(a_Size + a_Offset <= m_Size);
    // Perform a CPU -> GPU cudaMemcpy 
    CudaCheck(cudaMemcpy(reinterpret_cast<void*>(reinterpret_cast<CUdeviceptr>(m_DevPtr) + a_Offset), a_Data, a_Size, cudaMemcpyKind::cudaMemcpyHostToDevice));
};

void MemoryBuffer::Read(void* a_Dst, size_t a_ReadSize, size_t a_SrcOffset) const
{
    // Assert that the offset won't cause us to go outside of the allocated memory
    assert(a_SrcOffset + a_ReadSize <= m_Size);
    // Perform a GPU -> CPU cudaMemcpy 
    CudaCheck(cudaMemcpy(a_Dst, reinterpret_cast<void*>(reinterpret_cast<CUdeviceptr>(m_DevPtr) + a_SrcOffset), a_ReadSize, cudaMemcpyDeviceToHost));
};

void MemoryBuffer::CopyFrom(MemoryBuffer& a_MemoryBuffer, size_t a_Size, size_t a_DstOffset, size_t a_SrcOffset)
{
    // Assert that the offsets won't cause us to go outside of the allocated memory
    assert(a_DstOffset + a_Size <= m_Size);
    assert(a_SrcOffset + a_Size <= a_MemoryBuffer.GetSize());

    // Perform a GPU -> GPU cudaMemcpy 
    CudaCheck(cudaMemcpy(reinterpret_cast<void*>(reinterpret_cast<CUdeviceptr>(m_DevPtr) + a_DstOffset), reinterpret_cast<void*>(*a_MemoryBuffer + a_SrcOffset),
        a_Size, cudaMemcpyDeviceToDevice));
}

void MemoryBuffer::Resize(size_t a_NewSize)
{
    // Free the previously allocated memory if such exists
    if (m_DevPtr)
        cudaFree(m_DevPtr);

    // Allocate new memory of the requested size
    auto result = cudaMalloc(&m_DevPtr, a_NewSize);

    CudaCheck(result);

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
