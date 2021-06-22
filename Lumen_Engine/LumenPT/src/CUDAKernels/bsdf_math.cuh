#pragma once

/*
 * This file contains some BRDF helper functions and math related things.
 * It's essentially here to contain anything that is shared by multiple BRDF related files.
 */
#include <cuda_runtime.h>
#include <sutil/vec_math.h>
//#include <math.h>
//#include "../../vendor/Include/Cuda/cuda/helpers.h"

#define sqr(a) ((a)*(a))
#define REFERENCE_OF(x) x&
#define CONSTREF_OF(x) const x&

__device__ __forceinline__ float lerp(const float a, const float b, const float t)
{
	return a + t * (b - a);
}

#ifndef PI
#define PI					3.14159265358979323846264f
#endif
#ifndef INVPI
#define INVPI				0.31830988618379067153777f
#endif
#ifndef INV2PI
#define INV2PI				0.15915494309189533576888f
#endif
#ifndef TWOPI
#define TWOPI				6.28318530717958647692528f
#endif
#ifndef SQRT_PI_INV
#define SQRT_PI_INV			0.56418958355f
#endif
#ifndef LARGE_FLOAT
#define LARGE_FLOAT			1e34f
#endif
#ifndef EPSILON
#define EPSILON				0.0001f
#endif
#ifndef MINROUGHNESS
#define MINROUGHNESS		0.0001f	// minimal GGX roughness
#endif
#ifndef BLACK
#define BLACK				make_float3( 0 )
#endif
#ifndef WHITE
#define WHITE				make_float3( 1 )
#endif
#ifndef MIPLEVELCOUNT
#define MIPLEVELCOUNT		5
#endif



__device__ __forceinline__ void SetupTangentSpace(const float3& N, float3& T, float3& B)
{
	// "Building an Orthonormal Basis, Revisited"
	float sign = copysignf(1.0f, N.z);
	const float a = -1.0f / (sign + N.z);
	const float b = N.x * N.y * a;
	B = make_float3(1.0f + sign * N.x * N.x * a, sign * b, -sign * N.x);
	T = make_float3(b, sign + N.y * N.y * a, -N.y);
}

__device__ __forceinline__ float3 World2Tangent(const float3& V, const float3& N)
{
	float sign = copysignf(1.0f, N.z);
	const float a = -1.0f / (sign + N.z);
	const float b = N.x * N.y * a;
	const float3 B = make_float3(1.0f + sign * N.x * N.x * a, sign * b, -sign * N.x);
	const float3 T = make_float3(b, sign + N.y * N.y * a, -N.y);
	return make_float3(dot(V, T), dot(V, B), dot(V, N));
}

__device__ __forceinline__ float3 World2Tangent(const float3& V, const float3& N, const float3& T, const float3& B)
{
	return make_float3(dot(V, T), dot(V, B), dot(V, N));
}

__device__ __forceinline__ float3 Tangent2World(const float3& V, const float3& N)
{
	float3 T, B;
	SetupTangentSpace(N, T, B);
	return V.x * T + V.y * B + V.z * N;
}

__device__ __forceinline__ float3 Tangent2World(const float3& V, const float3& N, const float3& T, const float3& B)
{
	return V.x * T + V.y * B + V.z * N;
}


#define DIFFWEIGHT	weights.x
#define SHEENWEIGHT	weights.y
#define SPECWEIGHT	weights.z
#define COATWEIGHT	weights.w

#define GGXMDF		1001
#define GTR1MDF		1002


/*
 * Some generic math defines.
 */
#define FLOAT3_1 make_float3(1.f, 1.f, 1.f)
#define FLOAT3_0 make_float3(0.f, 0.f, 0.f)
#define FLOAT3_X make_float3(1.f, 0.f, 0.f)
#define FLOAT3_Y make_float3(0.f, 1.f, 0.f)
#define FLOAT3_Z make_float3(0.f, 0.f, 1.f)

 /*
  * Macros
  */
#define Square(X) ((X) * (X))
#define AbsDot(X, Y) fabsf(dot((X), (Y)))

  /*
   * Generic math functions directly taken from: https://www.pbr-book.org/3ed-2018/Reflection_Models
   */
__device__ __forceinline__ float CosTheta(const float3& w) { return w.z; }
__device__ __forceinline__ float Cos2Theta(const float3& w) { return w.z * w.z; }
__device__ __forceinline__ float Sin2Theta(const float3& w) { return fmaxf(0.f, 1.f - Cos2Theta(w)); }
__device__ __forceinline__ float SinTheta(const float3& w) { return sqrtf(Sin2Theta(w)); }
__device__ __forceinline__ float AbsCosTheta(const float3& w) { return fabsf(w.z); }
__device__ __forceinline__ float CosPhi(const float3& w) { float sinTheta = SinTheta(w); return (sinTheta == 0) ? 1 : clamp(w.x / sinTheta, -1.f, 1.f); }
__device__ __forceinline__ float SinPhi(const float3& w) { float sinTheta = SinTheta(w); return (sinTheta == 0) ? 0 : clamp(w.y / sinTheta, -1.f, 1.f); }
__device__ __forceinline__ float Cos2Phi(const float3& w) { return CosPhi(w) * CosPhi(w); }
__device__ __forceinline__ float Sin2Phi(const float3& w) { return SinPhi(w) * SinPhi(w); }
__device__ __forceinline__ float TanTheta(const float3& w) { return SinTheta(w) / CosTheta(w); }
__device__ __forceinline__ float Tan2Theta(const float3& w) { return Sin2Theta(w) / Cos2Theta(w); }


__device__ __forceinline__ float3 DiffuseReflectionCosWeighted(const float r0, const float r1)
{
	const float term1 = TWOPI * r0, term2 = sqrtf(1 - r1);
	float s, c;
	sincosf(term1, &s, &c);
	return make_float3(c * term2, s * term2, sqrtf(r1));
}

__device__ __forceinline__ float3 DiffuseReflectionCosWeighted(const float r0, const float r1, const float3& N, const float3& T, const float3& B)
{
	const float term1 = TWOPI * r0, term2 = sqrtf(1 - r1);
	float s, c;
	sincosf(term1, &s, &c);
	return (c * term2 * T) + (s * term2) * B + sqrtf(r1) * N;
}

__device__ __forceinline__ float3 DiffuseReflectionUniform(const float r0, const float r1)
{
	const float term1 = TWOPI * r0, term2 = sqrtf(1 - r1 * r1);
	float s, c;
	sincosf(term1, &s, &c);
	return make_float3(c * term2, s * term2, r1);
}

__device__ __forceinline__ float3 UniformSampleSphere(const float r0, const float r1)
{
	const float z = 1.0f - 2.0f * r1; // [-1~1]
	const float term1 = TWOPI * r0, term2 = sqrtf(1 - z * z);
	float s, c;
	sincosf(term1, &s, &c);
	return make_float3(c * term2, s * term2, z);
}

__device__ __forceinline__ float3 UniformSampleCone(const float r0, const float r1, const float cos_outer)
{
	float cosTheta = 1.0f - r1 + r1 * cos_outer;
	float term2 = sqrtf(1 - cosTheta * cosTheta);
	const float term1 = TWOPI * r0;
	float s, c;
	sincosf(term1, &s, &c);
	return make_float3(c * term2, s * term2, cosTheta);
}

__device__ __forceinline__ float3 CatmullRom(const float3& p0, const float3& p1, const float3& p2, const float3& p3, const float t)
{
	const float3 a = 2 * p1;
	const float3 b = p2 - p0;
	const float3 c = 2 * p0 - 5 * p1 + 4 * p2 - p3;
	const float3 d = -1 * p0 + 3 * p1 - 3 * p2 + p3;
	return 0.5f * (a + (b * t) + (c * t * t) + (d * t * t * t));
}
