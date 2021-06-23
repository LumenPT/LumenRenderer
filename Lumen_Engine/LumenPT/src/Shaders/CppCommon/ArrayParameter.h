#pragma once
#include <cuda_runtime.h>
#include <array>

template<typename T, size_t N>
struct ArrayParameter
{

    __host__ __device__
    ArrayParameter() = default;

    __host__
    ArrayParameter(const std::array<T,N>& a_Copied)
        :
    m_Elements()
    {
        std::copy(a_Copied.begin(), a_Copied.end(), m_Elements);
    }

    __host__
    ArrayParameter& operator=(const std::array<T, N>& a_Copied)
    {
        std::copy(a_Copied.begin(), a_Copied.end(), m_Elements);
        return *this;
    }

    __host__ __device__
    T& operator[](size_t a_Index)
    {
        assert(a_Index < N && "Index out of range!");

        return m_Elements[a_Index];

    }

    __host__ __device__
    const T& operator[](size_t a_Index) const
    {
        assert(a_Index < N && "Index out of range!");

        return m_Elements[a_Index];

    }

    T m_Elements[N];

};