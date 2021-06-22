#pragma once

#include <cuda_runtime.h>

//#define __CUDACC__ //TODO: temp defined
#include <cuda_fp16.h>
#include <vector_types.h>

struct half4
{

    __host__ __device__
    half4()
    {
        m_Elements[0] = half2{ __float2half(0.f), __float2half(0.f) };
        m_Elements[1] = half2{ __float2half(0.f), __float2half(0.f) };
    }

    __host__ __device__
    half4(float a_Val)
    {
        m_Elements[0] = half2{ __float2half(a_Val), __float2half(a_Val) };
        m_Elements[1] = half2{ __float2half(a_Val), __float2half(a_Val) };
    }

    __host__ __device__
    half4(float a_V1, float a_V2, float a_V3, float a_V4)
    {
        m_Elements[0] = half2{ __float2half(a_V1), __float2half(a_V2) };
        m_Elements[1] = half2{ __float2half(a_V3), __float2half(a_V4) };
    }

    __host__ __device__
    half4(const float4& a_Vec)
    {
        m_Elements[0] = half2{ __float2half(a_Vec.x), __float2half(a_Vec.y) };
        m_Elements[1] = half2{ __float2half(a_Vec.z), __float2half(a_Vec.w) };
    }

    __host__ __device__
    half4(const float3& a_Vec, float a_Val)
    {
        m_Elements[0] = half2{ __float2half(a_Vec.x), __float2half(a_Vec.y) };
        m_Elements[1] = half2{ __float2half(a_Vec.z), __float2half(a_Val) };
    }

    __host__ __device__
    half4(const float2& a_Vec1, const float2& a_Vec2)
    {
        m_Elements[0] = half2{ __float2half(a_Vec1.x), __float2half(a_Vec1.y) };
        m_Elements[1] = half2{ __float2half(a_Vec2.x), __float2half(a_Vec2.y) };
    }

    __host__ __device__
    half4(const half& a_V1, const half& a_V2, const half& a_V3, const half& a_V4)
    {
        m_Elements[0] = half2{ a_V1, a_V2 };
        m_Elements[1] = half2{ a_V3, a_V4 };
    }

    __host__ __device__
    half4(const half2& a_Half2A, const half2& a_Half2B)
    {
        m_Elements[0] = a_Half2A;
        m_Elements[1] = a_Half2B;
    }

    __inline__ __device__
    half4(const ushort4& a_Ushort4)
    {
        m_Elements[0] = half2{ __ushort_as_half(a_Ushort4.x), __ushort_as_half(a_Ushort4.y) };
        m_Elements[1] = half2{ __ushort_as_half(a_Ushort4.z), __ushort_as_half(a_Ushort4.w) };
    }

    __host__ __device__
    float4 AsFloat4() const
    {
        return float4{
            __half2float(m_Elements[0].x),
            __half2float(m_Elements[0].y),
            __half2float(m_Elements[1].x),
            __half2float(m_Elements[1].y)
        };
    }

    __inline__ __device__
    ushort4 AsUshort4() const
    {
        return ushort4
        {
            __half_as_ushort(m_Elements[0].x),
            __half_as_ushort(m_Elements[0].y),
            __half_as_ushort(m_Elements[1].x),
            __half_as_ushort(m_Elements[1].y)
        };
    }

    half2 m_Elements[2];

};

#if defined(__CUDACC__)

__forceinline__ __device__
half4 operator+(const half4& a_Left, const half4& a_Right)
{
    return half4
    {
        __hadd2(a_Left.m_Elements[0], a_Right.m_Elements[0]),
        __hadd2(a_Left.m_Elements[1], a_Right.m_Elements[1])
    };
}

__forceinline__ __device__
void operator+=(half4& a_Left, const half4& a_Right)
{
    a_Left.m_Elements[0] = __hadd2(a_Left.m_Elements[0], a_Right.m_Elements[0]);
    a_Left.m_Elements[1] = __hadd2(a_Left.m_Elements[1], a_Right.m_Elements[1]);
}

__forceinline__ __device__
half4 operator*(const half4& a_Left, const half4& a_Right)
{
    return half4
    {
        __hmul2(a_Left.m_Elements[0], a_Right.m_Elements[0]),
        __hmul2(a_Left.m_Elements[1], a_Right.m_Elements[1])
    };
}

__forceinline__ __device__
half4 operator*(const half4& a_Left, const float& a_Right)
{
    const half2 scalar2{ __float2half(a_Right), __float2half(a_Right) };

    return half4
    {
        __hmul2(a_Left.m_Elements[0], scalar2),
        __hmul2(a_Left.m_Elements[1], scalar2)
    };
}

__forceinline__ __device__
void operator*=(half4& a_Left, const half4& a_Right)
{
    a_Left.m_Elements[0] = __hmul2(a_Left.m_Elements[0], a_Right.m_Elements[0]);
    a_Left.m_Elements[1] = __hmul2(a_Left.m_Elements[1], a_Right.m_Elements[1]);
}

__forceinline__ __device__
void operator*=(half4& a_Left, const float& a_Right)
{
    const half2 scalar2{ __float2half(a_Right), __float2half(a_Right) };

    a_Left.m_Elements[0] = __hmul2(a_Left.m_Elements[0], scalar2);
    a_Left.m_Elements[1] = __hmul2(a_Left.m_Elements[1], scalar2);
}

__forceinline__ __device__
half4 operator/(const half4& a_Left, const half4& a_Right)
{
    return half4
    {
        __h2div(a_Left.m_Elements[0], a_Right.m_Elements[0]),
        __h2div(a_Left.m_Elements[1], a_Right.m_Elements[1])
    };
}

__forceinline__ __device__
half4 operator/(const half4& a_Left, const float& a_Right)
{
    const half2 scalar2{ __float2half(a_Right), __float2half(a_Right) };

    return half4
    {
        __h2div(a_Left.m_Elements[0], scalar2),
        __h2div(a_Left.m_Elements[1], scalar2)
    };
}

__forceinline__ __device__
void operator/=(half4& a_Left, const half4& a_Right)
{
    a_Left.m_Elements[0] = __h2div(a_Left.m_Elements[0], a_Right.m_Elements[0]);
    a_Left.m_Elements[1] = __h2div(a_Left.m_Elements[1], a_Right.m_Elements[1]);
}

__forceinline__ __device__
void operator/=(half4& a_Left, const float& a_Right)
{
    const half2 scalar2{ __float2half(a_Right), __float2half(a_Right) };

    a_Left.m_Elements[0] = __h2div(a_Left.m_Elements[0], scalar2);
    a_Left.m_Elements[1] = __h2div(a_Left.m_Elements[1], scalar2);
}

#endif



union half4Ushort4
{
    half4 m_Half4;
    ushort4 m_Ushort4;
};