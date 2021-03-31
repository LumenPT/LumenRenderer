//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
#pragma once

#include <sutil/Matrix.h>

//------------------------------------------------------------------------------
//
// Quaternion class
//
//------------------------------------------------------------------------------

namespace sutil
{

class Quaternion
{
public:
    SUTIL_INLINE SUTIL_HOSTDEVICE Quaternion()
    { q[0] = q[1] = q[2] = q[3] = 0.0; }

    SUTIL_INLINE SUTIL_HOSTDEVICE Quaternion( float w, float x, float y, float z )
    { q[0] = w; q[1] = x; q[2] = y; q[3] = z; }

    SUTIL_INLINE SUTIL_HOSTDEVICE Quaternion( const float3& from, const float3& to );

    SUTIL_INLINE SUTIL_HOSTDEVICE Quaternion( const Quaternion& a )
    { q[0] = a[0];  q[1] = a[1];  q[2] = a[2];  q[3] = a[3]; }

    SUTIL_INLINE SUTIL_HOSTDEVICE Quaternion ( float angle, const float3& axis );

    // getters and setters
    SUTIL_INLINE SUTIL_HOSTDEVICE void setW(float _w)       { q[0] = _w; }
    SUTIL_INLINE SUTIL_HOSTDEVICE void setX(float _x)       { q[1] = _x; }
    SUTIL_INLINE SUTIL_HOSTDEVICE void setY(float _y)       { q[2] = _y; }
    SUTIL_INLINE SUTIL_HOSTDEVICE void setZ(float _z)       { q[3] = _z; }
    SUTIL_INLINE SUTIL_HOSTDEVICE float w() const           { return q[0]; }
    SUTIL_INLINE SUTIL_HOSTDEVICE float x() const           { return q[1]; }
    SUTIL_INLINE SUTIL_HOSTDEVICE float y() const           { return q[2]; }
    SUTIL_INLINE SUTIL_HOSTDEVICE float z() const           { return q[3]; }


    SUTIL_INLINE SUTIL_HOSTDEVICE Quaternion& operator-=(const Quaternion& r)
    { q[0] -= r[0]; q[1] -= r[1]; q[2] -= r[2]; q[3] -= r[3]; return *this; }

    SUTIL_INLINE SUTIL_HOSTDEVICE Quaternion& operator+=(const Quaternion& r)
    { q[0] += r[0]; q[1] += r[1]; q[2] += r[2]; q[3] += r[3]; return *this; }

    SUTIL_INLINE SUTIL_HOSTDEVICE Quaternion& operator*=(const Quaternion& r);

    SUTIL_INLINE SUTIL_HOSTDEVICE Quaternion& operator/=(const float a);

    SUTIL_INLINE SUTIL_HOSTDEVICE Quaternion conjugate()
    { return Quaternion( q[0], -q[1], -q[2], -q[3] ); }

    SUTIL_INLINE SUTIL_HOSTDEVICE void rotation( float& angle, float3& axis ) const;
    SUTIL_INLINE SUTIL_HOSTDEVICE void rotation( float& angle, float& x, float& y, float& z ) const;
    SUTIL_INLINE SUTIL_HOSTDEVICE Matrix4x4 rotationMatrix() const;
    SUTIL_INLINE SUTIL_HOSTDEVICE Matrix3x3 ToMat3() const;

    SUTIL_INLINE SUTIL_HOSTDEVICE float& operator[](int i)      { return q[i]; }
    SUTIL_INLINE SUTIL_HOSTDEVICE float operator[](int i)const  { return q[i]; }

    // l2 norm
    SUTIL_INLINE SUTIL_HOSTDEVICE float norm() const
    { return sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]); }

    SUTIL_INLINE SUTIL_HOSTDEVICE float  normalize();

private:
    float q[4];
};


SUTIL_INLINE SUTIL_HOSTDEVICE  Quaternion::Quaternion( const float3& from, const float3& to )
{
    const float3 c = cross( from, to );
    q[0] = dot(from, to);
    q[1] = c.x;
    q[2] = c.y;
    q[3] = c.z;
    
}


SUTIL_INLINE SUTIL_HOSTDEVICE  Quaternion::Quaternion( float angle, const float3&  axis )
{
    const float  n        = length( axis );
    const float  inverse  = 1.0f/n;
    const float3 naxis    = axis*inverse;
    const float  s        = sinf(angle/2.0f);

    q[0] = naxis.x*s*inverse;
    q[1] = naxis.y*s*inverse;
    q[2] = naxis.z*s*inverse;
    q[3] = cosf(angle/2.0f);
}


SUTIL_INLINE SUTIL_HOSTDEVICE  Quaternion& Quaternion::operator*=(const Quaternion& r)
{

    float w = q[0]*r[0] - q[1]*r[1] - q[2]*r[2] - q[3]*r[3];
    float x = q[0]*r[1] + q[1]*r[0] + q[2]*r[3] - q[3]*r[2];
    float y = q[0]*r[2] + q[2]*r[0] + q[3]*r[1] - q[1]*r[3];
    float z = q[0]*r[3] + q[3]*r[0] + q[1]*r[2] - q[2]*r[1];

    q[0] = w;
    q[1] = x;
    q[2] = y;
    q[3] = z;
    return *this;
}


SUTIL_INLINE SUTIL_HOSTDEVICE  Quaternion& Quaternion::operator/=(const float a)
{
    float inverse = 1.0f/a;
    q[0] *= inverse;
    q[1] *= inverse;
    q[2] *= inverse;
    q[3] *= inverse;
    return *this;
}

SUTIL_INLINE SUTIL_HOSTDEVICE  void Quaternion::rotation( float& angle, float3& axis ) const
{
    Quaternion n = *this;
    n.normalize();
    axis.x = n[1];
    axis.y = n[2];
    axis.z = n[3];
    angle = 2.0f * acosf(n[0]);
}

SUTIL_INLINE SUTIL_HOSTDEVICE  void Quaternion::rotation(
        float& angle,
        float& x,
        float& y,
        float& z
        ) const
{
    Quaternion n = *this;
    n.normalize();
    x = n[1];
    y = n[2];
    z = n[3];
    angle = 2.0f * acosf(n[0]);
}

SUTIL_INLINE SUTIL_HOSTDEVICE  float Quaternion::normalize()
{
    float n = norm();
    float inverse = 1.0f/n;
    q[0] *= inverse;
    q[1] *= inverse;
    q[2] *= inverse;
    q[3] *= inverse;
    return n;
}


SUTIL_INLINE SUTIL_HOSTDEVICE  Quaternion operator*(const float a, const Quaternion &r)
{ return Quaternion(a*r[0], a*r[1], a*r[2], a*r[3]); }


SUTIL_INLINE SUTIL_HOSTDEVICE  Quaternion operator*(const Quaternion &r, const float a)
{ return Quaternion(a*r[0], a*r[1], a*r[2], a*r[3]); }


SUTIL_INLINE SUTIL_HOSTDEVICE  Quaternion operator/(const Quaternion &r, const float a)
{
    float inverse = 1.0f/a;
    return Quaternion( r[0]*inverse, r[1]*inverse, r[2]*inverse, r[3]*inverse);
}


SUTIL_INLINE SUTIL_HOSTDEVICE  Quaternion operator/(const float a, const Quaternion &r)
{
    float inverse = 1.0f/a;
    return Quaternion( r[0]*inverse, r[1]*inverse, r[2]*inverse, r[3]*inverse);
}


SUTIL_INLINE SUTIL_HOSTDEVICE  Quaternion operator-(const Quaternion& l, const Quaternion& r)
{ return Quaternion(l[0]-r[0], l[1]-r[1], l[2]-r[2], l[3]-r[3]); }


SUTIL_INLINE SUTIL_HOSTDEVICE  bool operator==(const Quaternion& l, const Quaternion& r)
{ return ( l[0] == r[0] && l[1] == r[1] && l[2] == r[2] && l[3] == r[3] ); }


SUTIL_INLINE SUTIL_HOSTDEVICE  bool operator!=(const Quaternion& l, const Quaternion& r)
{ return !(l == r); }


SUTIL_INLINE SUTIL_HOSTDEVICE  Quaternion operator+(const Quaternion& l, const Quaternion& r)
{ return Quaternion(l[0]+r[0], l[1]+r[1], l[2]+r[2], l[3]+r[3]); }


SUTIL_INLINE SUTIL_HOSTDEVICE  Quaternion operator*(const Quaternion& l, const Quaternion& r)
{
    float w = l[0]*r[0] - l[1]*r[1] - l[2]*r[2] - l[3]*r[3];
    float x = l[0]*r[1] + l[1]*r[0] + l[2]*r[3] - l[3]*r[2];
    float y = l[0]*r[2] + l[2]*r[0] + l[3]*r[1] - l[1]*r[3];
    float z = l[0]*r[3] + l[3]*r[0] + l[1]*r[2] - l[2]*r[1];
    return Quaternion( w, x, y, z );
}

SUTIL_INLINE SUTIL_HOSTDEVICE  float dot( const Quaternion& l, const Quaternion& r )
{
    return l.w()*r.w() + l.x()*r.x() + l.y()*r.y() + l.z()*r.z();
}


SUTIL_INLINE SUTIL_HOSTDEVICE  Matrix4x4 Quaternion::rotationMatrix() const
{
    Matrix4x4 m;

    const float qw = q[0];
    const float qx = q[1];
    const float qy = q[2];
    const float qz = q[3];

    m[0*4+0] = 1.0f - 2.0f*qy*qy - 2.0f*qz*qz;
    m[0*4+1] = 2.0f*qx*qy - 2.0f*qz*qw;
    m[0*4+2] = 2.0f*qx*qz + 2.0f*qy*qw;
    m[0*4+3] = 0.0f;

    m[1*4+0] = 2.0f*qx*qy + 2.0f*qz*qw;
    m[1*4+1] = 1.0f - 2.0f*qx*qx - 2.0f*qz*qz;
    m[1*4+2] = 2.0f*qy*qz - 2.0f*qx*qw;
    m[1*4+3] = 0.0f;

    m[2*4+0] = 2.0f*qx*qz - 2.0f*qy*qw;
    m[2*4+1] = 2.0f*qy*qz + 2.0f*qx*qw;
    m[2*4+2] = 1.0f - 2.0f*qx*qx - 2.0f*qy*qy;
    m[2*4+3] = 0.0f;

    m[3*4+0] = 0.0f;
    m[3*4+1] = 0.0f;
    m[3*4+2] = 0.0f;
    m[3*4+3] = 1.0f;

    return m;
}

SUTIL_INLINE SUTIL_HOSTDEVICE  Matrix3x3 Quaternion::ToMat3() const
{
    Matrix3x3 Result;
    float qxx(q[1] * q[1]);
    float qyy(q[2] * q[2]);
    float qzz(q[3] * q[3]);
    float qxz(q[1] * q[3]);
    float qxy(q[1] * q[2]);
    float qyz(q[2] * q[3]);
    float qwx(q[0] * q[1]);
    float qwy(q[0] * q[2]);
    float qwz(q[0] * q[3]);

    Result[0 * 3 + 0] = 1.f - 2.f * (qyy + qzz);
    Result[1 * 3 + 0] = 2.f * (qxy + qwz);
    Result[2 * 3 + 0] = 2.f * (qxz - qwy);

    Result[0 * 3 + 1] = 2.f * (qxy - qwz);
    Result[1 * 3 + 1] = 1.f - 2.f * (qxx + qzz);
    Result[2 * 3 + 1] = 2.f * (qyz + qwx);

    Result[0 * 3 + 2] = 2.f * (qxz + qwy);
    Result[1 * 3 + 2] = 2.f * (qyz - qwx);
    Result[2 * 3 + 2] = 1.f - 2.f * (qxx + qyy);

    return Result;
}

} // end namespace sutil
