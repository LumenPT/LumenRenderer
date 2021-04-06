#pragma once

#include <type_traits>
#include <vector>

// Wrapper class to allow easier work with GPU memory without having to worry about GPU memory leaks
class MemoryBuffer
{
public:

    // Create a memory buffer of the specified size
    MemoryBuffer(size_t a_Size = 0);
    // Create a memory buffer to contain the provided std::vector's data
    template<typename ElementType>
    MemoryBuffer(std::vector<ElementType>& a_VectorData);
    ~MemoryBuffer();

    // Movement constructor
    MemoryBuffer(MemoryBuffer&& a_Other);

    // Dereferencing the buffer returns a CUdeviceptr which is what OptiX primarily uses
    unsigned long long& operator*();

    // Take an existing CPU struct and write it to the memory.
    template<typename DataType, typename Enabler = std::enable_if_t<!std::is_pointer<DataType>::value>>
    void Write(const DataType& a_Data, size_t a_Offset = 0)
    {
        Write(&a_Data, sizeof(DataType), a_Offset);
    }

    // Copy from CPU data to the memory buffer
    void Write(const void* a_Data, size_t a_Size, size_t a_Offset = 0);

    // Readback from the GPU memory to the destination CPU memory
    void Read(void* a_Dst, size_t a_ReadSize, size_t a_SrcOffset) const;

    // Copy from an existing memory buffer
    void CopyFrom(MemoryBuffer& a_MemoryBuffer, size_t a_Size, size_t a_DstOffset = 0, size_t a_SrcOffset = 0);

    void Resize(size_t a_NewSize);

    // Returns the size of the buffer in bytes
    size_t GetSize() const; 

    // Return a GPU pointer to the start of the buffer. The pointer is of the specified template type
    template<typename PointerType = void>
    PointerType* GetDevicePtr() const { return static_cast<PointerType*>(m_DevPtr); }    

    // Return CUdeviceptr to the start of buffer
    unsigned long long GetCUDAPtr() const;

private:
    // The CUdeviceptr is essentially a uint64_t representation of a void*, so making a union of the two simplifies managing them
    union
    {
        unsigned long long m_CudaPtr;
        void* m_DevPtr;        
    };
    size_t m_Size; // Size of the buffer in bytes
};

template <typename ElementType>
MemoryBuffer::MemoryBuffer(std::vector<ElementType>& a_VectorData)
    : m_DevPtr(0)
{
    // Resize the buffer to fit the entirety of the vector
    Resize(a_VectorData.size() * sizeof(ElementType));

    // Copy the data from the vector to the buffer's memory
    Write(a_VectorData.data(), a_VectorData.size() * sizeof(ElementType));
}
