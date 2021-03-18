#pragma once

#include <type_traits>
#include <vector>

class MemoryBuffer
{
public:

    MemoryBuffer(size_t a_Size = 0);
    template<typename ElementType>
    MemoryBuffer(std::vector<ElementType>& a_VectorData);
    ~MemoryBuffer();

    MemoryBuffer(MemoryBuffer&& a_Other);

    unsigned long long& operator*();

    template<typename DataType, typename Enabler = std::enable_if_t<!std::is_pointer<DataType>::value>>
    void Write(const DataType& a_Data, size_t a_Offset = 0)
    {
        Write(&a_Data, sizeof(DataType), a_Offset);
    }

    void Write(const void* a_Data, size_t a_Size, size_t a_Offset = 0);

    void Read(void* a_Dst, size_t a_ReadSize, size_t a_SrcOffset) const;

    void CopyFrom(MemoryBuffer& a_MemoryBuffer, size_t a_Size, size_t a_DstOffset = 0, size_t a_SrcOffset = 0);

    void Resize(size_t a_NewSize);

    size_t GetSize() const;

    template<typename PointerType = void>
    PointerType* GetDevicePtr() const { return static_cast<PointerType*>(m_DevPtr); }

    

    unsigned long long GetCUDAPtr() const;

private:
    union
    {
        unsigned long long m_CudaPtr;
        void* m_DevPtr;        
    };
    size_t m_Size;
};

template <typename ElementType>
MemoryBuffer::MemoryBuffer(std::vector<ElementType>& a_VectorData)
    : m_DevPtr(0)
{
    Resize(a_VectorData.size() * sizeof(ElementType));

    Write(a_VectorData.data(), a_VectorData.size() * sizeof(ElementType));
}
