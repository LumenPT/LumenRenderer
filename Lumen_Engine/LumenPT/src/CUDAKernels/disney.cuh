#pragma once

#include "bsdf_math.cuh"
#include "ggxmdf.cuh"
#include "frosted.cuh"

/*
 * The below code has been copied from LightHouse2: https://github.com/jbikker/lighthouse2/blob/master/lib/RenderCore_Optix7Filter/kernels/bsdfs/disney.h
 */

/*
# Copyright Disney Enterprises, Inc.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License
# and the following modification to it: Section 6 Trademarks.
# deleted and replaced with:
#
# 6. Trademarks. This License does not grant permission to use the
# trade names, trademarks, service marks, or product names of the
# Licensor and its affiliates, except as required for reproducing
# the content of the NOTICE file.
#
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Adapted to C++ by Miles Macklin 2016
*/

enum BSDFType
{
	eReflected,
	eTransmitted,
	eSpecular
};

struct ShadingData
{
	float3 color;			//The albedo surface color.

	union
	{
		uint4 parameters;
		unsigned char paramsAsChar[16];	//Aliasing to char.
		float paramsAsFloat[4];			//Aliasing to float
	};
	/* 16 uchars:   x: 0..7 = metallic, 8..15 = subsurface, 16..23 = specular, 24..31 = roughness;
					y: 0..7 = specTint, 8..15 = anisotropic, 16..23 = sheen, 24..31 = sheenTint;
					z: 0..7 = clearcoat, 8..15 = clearcoatGloss, 16..23 = transmission, 24..31 = dummy;
					w: eta (32-bit float). */

#define SET_METALLIC(X) (shadingData.paramsAsChar[0] = (static_cast<unsigned char>((X) * static_cast<unsigned char>(255))))
#define SET_SUBSURFACE(X) (shadingData.paramsAsChar[1] = (static_cast<unsigned char>((X) * static_cast<unsigned char>(255))))
#define SET_SPECULAR(X) (shadingData.paramsAsChar[2] = (static_cast<unsigned char>((X) * static_cast<unsigned char>(255))))
#define SET_ROUGHNESS(X) (shadingData.paramsAsChar[3] = (static_cast<unsigned char>((X) * static_cast<unsigned char>(255))))

#define SET_SPECTINT(X) (shadingData.paramsAsChar[4] = (X))

#define SET_CLEARCOAT(X) (shadingData.paramsAsChar[8] = (static_cast<unsigned char>((X) * static_cast<unsigned char>(255))))
#define SET_CLEARCOATGLOSS(X) (shadingData.paramsAsChar[9] = (static_cast<unsigned char>((X) * static_cast<unsigned char>(255))))
#define SET_TRANSMISSION(X) (shadingData.paramsAsChar[10] = (static_cast<unsigned char>((X) * static_cast<unsigned char>(255))))

#define SET_ETA(X) (shadingData.paramsAsFloat[3] = (X))


#define METALLIC CHAR2FLT( shadingData.parameters.x, 0 )
#define SUBSURFACE CHAR2FLT( shadingData.parameters.x, 8 )
#define SPECULAR CHAR2FLT( shadingData.parameters.x, 16 )
#define ROUGHNESS (max( 0.001f, CHAR2FLT( shadingData.parameters.x, 24 ) ))
#define SPECTINT CHAR2FLT( shadingData.parameters.y, 0 )
#define CLEARCOAT CHAR2FLT( shadingData.parameters.z, 0 )
#define CLEARCOATGLOSS CHAR2FLT( shadingData.parameters.z, 8 )
#define TRANSMISSION CHAR2FLT( shadingData.parameters.z, 16 )
#define ETA __uint_as_float( shadingData.parameters.w )


//float3 transmittance;
//float4 tint;		
//	__device__ int IsSpecular(const int layer) const { return 0; /* for now. */ }
//	__device__ bool IsEmissive() const { return color.x > 1.0f || color.y > 1.0f || color.z > 1.0f; }
//	__device__ void InvertETA() { parameters.w = __float_as_uint(1.0f / __uint_as_float(parameters.w)); }
//#define ANISOTROPIC CHAR2FLT( shadingData.parameters.y, 8 )
//#define SHEEN CHAR2FLT( shadingData.parameters.y, 16 )
//#define SHEENTINT CHAR2FLT( shadingData.parameters.y, 24 )
//#define TINT make_float3( shadingData.tint )
//#define LUMINANCE shadingData.tint.w
//#define DUMMY0 CHAR2FLT( shadingData.parameters.z, 24 )
};

__device__ static inline bool Refract(const float3& wi, const float3& n, const float eta, float3& wt)
{
	float cosThetaI = dot(n, wi);
	float sin2ThetaI = max(0.0f, 1.0f - cosThetaI * cosThetaI);
	float sin2ThetaT = eta * eta * sin2ThetaI;
	if (sin2ThetaT >= 1) return false; // TIR
	float cosThetaT = sqrtf(1.0f - sin2ThetaT);
	wt = eta * (wi * -1.0f) + (eta * cosThetaI - cosThetaT) * float3(n);
	return true;
}

__device__ static inline float SchlickFresnel(const float u)
{
	const float m = clamp(1 - u, 0.0f, 1.0f);
	return float(m * m) * (m * m) * m;
}

__device__ static inline float GTR1(const float NDotH, const float a)
{
	if (a >= 1) return INVPI;
	const float a2 = a * a;
	const float t = 1 + (a2 - 1) * NDotH * NDotH;
	return (a2 - 1) / (PI * logf(a2) * t);
}

__device__ static inline float GTR2(const float NDotH, const float a)
{
	const float a2 = a * a;
	const float t = 1.0f + (a2 - 1.0f) * NDotH * NDotH;
	return a2 / (PI * t * t);
}

__device__ static inline float SmithGGX(const float NDotv, const float alphaG)
{
	const float a = alphaG * alphaG;
	const float b = NDotv * NDotv;
	return 1 / (NDotv + sqrtf(a + b - a * b));
}

__device__ static float Fr(const float VDotN, const float eio)
{
	const float SinThetaT2 = sqr(eio) * (1.0f - VDotN * VDotN);
	if (SinThetaT2 > 1.0f) return 1.0f; // TIR
	const float LDotN = sqrtf(1.0f - SinThetaT2);
	// todo: reformulate to remove this division
	const float eta = 1.0f / eio;
	const float r1 = (VDotN - eta * LDotN) / (VDotN + eta * LDotN);
	const float r2 = (LDotN - eta * VDotN) / (LDotN + eta * VDotN);
	return 0.5f * (sqr(r1) + sqr(r2));
}

__device__ static inline float3 SafeNormalize(const float3& a)
{
	const float ls = dot(a, a);
	if (ls > 0.0f) return a * (1.0f / sqrtf(ls)); else return make_float3(0);
}

__device__ static float BSDFPdf(const ShadingData& shadingData, const float3& N, const float3& wo, const float3& wi)
{
	float bsdfPdf = 0.0f, brdfPdf;
	if (dot(wi, N) <= 0.0f) brdfPdf = INV2PI * SUBSURFACE * 0.5f; else
	{
		const float F = Fr(dot(N, wo), ETA);
		const float3 half = SafeNormalize(wi + wo);
		const float cosThetaHalf = abs(dot(half, N));
		const float pdfHalf = GTR2(cosThetaHalf, ROUGHNESS) * cosThetaHalf;
		// calculate pdf for each method given outgoing light vector
		const float pdfSpec = 0.25f * pdfHalf / max(1.e-6f, dot(wi, half));
		const float pdfDiff = abs(dot(wi, N)) * INVPI * (1.0f - SUBSURFACE);
		bsdfPdf = pdfSpec * F;
		brdfPdf = lerp(pdfDiff, pdfSpec, 0.5f);
	}
	return lerp(brdfPdf, bsdfPdf, TRANSMISSION);
}

// evaluate the BSDF for a given pair of directions
__device__ static float3 BSDFEval(const ShadingData& shadingData,
	const float3& N, const float3& wo, const float3& wi)
{
	const float NDotL = dot(N, wi);
	const float NDotV = dot(N, wo);
	const float3 H = normalize(wi + wo);
	const float NDotH = dot(N, H);
	const float LDotH = dot(wi, H);
	const float3 Cdlin = shadingData.color;
	const float Cdlum = .3f * Cdlin.x + .6f * Cdlin.y + .1f * Cdlin.z; // luminance approx.
	const float3 Ctint = Cdlum > 0.0f ? Cdlin / Cdlum : make_float3(1.0f); // normalize lum. to isolate hue+sat
	const float3 Cspec0 = lerp(SPECULAR * .08f * lerp(make_float3(1.0f), Ctint, SPECTINT), Cdlin, METALLIC);
	float3 bsdf = make_float3(0);
	float3 brdf = make_float3(0);
	if (TRANSMISSION > 0.0f)
	{
		// evaluate BSDF
		if (NDotL <= 0)
		{
			// transmission Fresnel
			const float F = Fr(NDotV, ETA);
			bsdf = make_float3((1.0f - F) / abs(NDotL) * (1.0f - METALLIC) * TRANSMISSION);
		}
		else
		{
			// specular lobe
			const float a = ROUGHNESS;
			const float Ds = GTR2(NDotH, a);

			// Fresnel term with the microfacet normal
			const float FH = Fr(LDotH, ETA);
			const float3 Fs = lerp(Cspec0, make_float3(1.0f), FH);
			const float Gs = SmithGGX(NDotV, a) * SmithGGX(NDotL, a);
			bsdf = (Gs * Ds) * Fs;
		}
	}
	if (TRANSMISSION < 1.0f)
	{
		// evaluate BRDF
		if (NDotL <= 0)
		{
			if (SUBSURFACE > 0.0f)
			{
				// take sqrt to account for entry/exit of the ray through the medium
				// this ensures transmitted light corresponds to the diffuse model
				const float3 s = make_float3(sqrtf(shadingData.color.x), sqrtf(shadingData.color.y), sqrtf(shadingData.color.z));
				const float FL = SchlickFresnel(abs(NDotL)), FV = SchlickFresnel(NDotV);
				const float Fd = (1.0f - 0.5f * FL) * (1.0f - 0.5f * FV);
				brdf = INVPI * s * SUBSURFACE * Fd * (1.0f - METALLIC);
			}
		}
		else
		{
			// specular
			const float a = ROUGHNESS;
			const float Ds = GTR2(NDotH, a);

			// Fresnel term with the microfacet normal
			const float FH = SchlickFresnel(LDotH);
			const float3 Fs = lerp(Cspec0, make_float3(1), FH);
			const float Gs = SmithGGX(NDotV, a) * SmithGGX(NDotL, a);

			// Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
			// and mix in diffuse retro-reflection based on roughness
			const float FL = SchlickFresnel(NDotL), FV = SchlickFresnel(NDotV);
			const float Fd90 = 0.5 + 2.0f * LDotH * LDotH * a;
			const float Fd = lerp(1.0f, Fd90, FL) * lerp(1.0f, Fd90, FV);

			// clearcoat (ior = 1.5 -> F0 = 0.04)
			const float Dr = GTR1(NDotH, lerp(.1, .001, CLEARCOATGLOSS));
			const float Fc = lerp(.04f, 1.0f, FH);
			const float Gr = SmithGGX(NDotL, .25) * SmithGGX(NDotV, .25);

			brdf = INVPI * Fd * Cdlin * (1.0f - METALLIC) * (1.0f - SUBSURFACE) + Gs * Fs * Ds + CLEARCOAT * Gr * Fc * Dr;
		}
	}

	return lerp(brdf, bsdf, TRANSMISSION);
}

// generate an importance sampled BSDF direction
__device__ static void BSDFSample(const ShadingData& shadingData,
	const float3& T, const float3& B, const float3& N, const float3& wo, float3& wi, float& pdf, BSDFType& type, const float r3, const float r4)
{
	if (r3 < TRANSMISSION)
	{
		// sample BSDF
		float F = Fr(dot(N, wo), ETA);
		if (r4 < F) // sample reflectance or transmission based on Fresnel term
		{
			// sample reflection
			const float r1 = r3 / TRANSMISSION;
			const float r2 = r4 / F;
			const float cosThetaHalf = sqrtf((1.0f - r2) / (1.0f + (sqr(ROUGHNESS) - 1.0f) * r2));
			const float sinThetaHalf = sqrtf(max(0.0f, 1.0f - sqr(cosThetaHalf)));
			float sinPhiHalf, cosPhiHalf;
			__sincosf(r1 * TWOPI, &sinPhiHalf, &cosPhiHalf);
			float3 half = T * (sinThetaHalf * cosPhiHalf) + B * (sinThetaHalf * sinPhiHalf) + N * cosThetaHalf;
			if (dot(half, wo) <= 0.0f) half *= -1.0f; // ensure half angle in same hemisphere as wo
			type = eReflected;
			wi = reflect(wo * -1.0f, half);
		}
		else // sample transmission
		{
			pdf = 0;
			if (Refract(wo, N, ETA, wi)) type = eSpecular, pdf = (1.0f - F) * TRANSMISSION;
			return;
		}
	}
	else // sample BRDF
	{
		const float r1 = (r3 - TRANSMISSION) / (1 - TRANSMISSION);
		if (r4 < 0.5f)
		{
			// sample diffuse	
			const float r2 = r4 * 2;
			float3 d;
			if (r2 < SUBSURFACE)
			{
				const float r5 = r2 / SUBSURFACE;
				d = DiffuseReflectionUniform(r1, r5), type = eTransmitted, d.z *= -1.0f;
			}
			else
			{
				const float r5 = (r2 - SUBSURFACE) / (1 - SUBSURFACE);
				d = DiffuseReflectionCosWeighted(r1, r5), type = eReflected;
			}
			wi = T * d.x + B * d.y + N * d.z;
		}
		else
		{
			// sample specular
			const float r2 = (r4 - 0.5f) * 2.0f;
			const float cosThetaHalf = sqrtf((1.0f - r2) / (1.0f + (sqr(ROUGHNESS) - 1.0f) * r2));
			const float sinThetaHalf = sqrtf(max(0.0f, 1.0f - sqr(cosThetaHalf)));
			float sinPhiHalf, cosPhiHalf;
			__sincosf(r1 * TWOPI, &sinPhiHalf, &cosPhiHalf);
			float3 half = T * (sinThetaHalf * cosPhiHalf) + B * (sinThetaHalf * sinPhiHalf) + N * cosThetaHalf;
			if (dot(half, wo) <= 0.0f) half *= -1.0f; // ensure half angle in same hemisphere as wi
			wi = reflect(wo * -1.0f, half);
			type = eReflected;
		}
	}
	pdf = BSDFPdf(shadingData, N, wo, wi);
}

// ----------------------------------------------------------------

/*
 * The below function evaluates the BSDF for a set of directions during a surface interaction.
 * Note: all directions point away from the surface.
 *
 * @ShadingData contains the surface properties such as metallic/roughness/sheen/transmission etc.
 * @iN is the normal of the surface.
 * @T is the tangent vector of the surface.
 * @wo is the incoming ray direction inverted.
 * @wi is the outgoing direction.
 * @pdf Represents the PDF of the evaluated BSDF.
 *
 * @returns the float3 BSDF.
 */
__device__ static float3 EvaluateBSDF(const ShadingData& shadingData, const float3& iN,
	const float3 wo, const float3 wi, float& pdf)
{
	const float3 bsdf = BSDFEval(shadingData, iN, wo, wi);
	pdf = BSDFPdf(shadingData, iN, wo, wi);
	return bsdf;
}

/*
 * The below function returns the BSDF for a surface interactions for an incoming.
 * Note: all directions point away from the surface.
 *
 * @shadingData contains the surface and material data.
 * @iN The surface normal.
 * @T The geometry surface tangent.
 * @wo The inverse incoming ray direction (surface to ray origin).
 * @r3 Random uniform float between 0 and 1 (inclusive).
 * @r4 Random uniform float between 0 and 1 (inclusive).
 *
 * @wi The generated outgoing direction.
 * @pdf The generated PDF for the sampled direction.
 *
 */
__device__ static float3 SampleBSDF(const ShadingData& shadingData,
	const float3& iN, const float3& T, const float3& wo,
	const float r3, const float r4, float3& wi, float& pdf)
{
	BSDFType type;
	const float3 B = normalize(cross(T, iN));
	const float3 Tfinal = cross(B, iN);
	BSDFSample(shadingData, Tfinal, B, iN, wo, wi, pdf, type, r3, r4);
	return BSDFEval(shadingData, iN, wo, wi);
}










/*
 * Below code is more complex and scary so I commented it out and pretend it doesn't exist so it can't hurt me.
 */

//#include "bsdf_math.cuh"
//#include "ggxmdf.cuh"
//#include "frosted.cuh"
//
//
///* disney2.h - License information:
// *
// * The below code was copied from the LightHouse2 repository on GitHub: https://github.com/jbikker/lighthouse2/blob/31c76490e01ccdcd01766ea046d66687375c04da/lib/sharedBSDFs/disney.h
//   In turn, LH2 has the below copyright notice:
//
//   This code has been adapted from AppleSeed: https://appleseedhq.net
//   The AppleSeed software is released under the MIT license.
//   Copyright (c) 2014-2018 Esteban Tovagliari, The appleseedhq Organization.
//
//   Permission is hereby granted, free of charge, to any person obtaining a copy
//   of this software and associated documentation files (the "Software"), to deal
//   in the Software without restriction, including without limitation the rights
//   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//   copies of the Software, and to permit persons to whom the Software is
//   furnished to do so, subject to the following conditions:
//
//   The above copyright notice and this permission notice shall be included in
//   all copies or substantial portions of the Software.
//
//   // https://github.com/appleseedhq/appleseed/blob/master/src/appleseed/renderer/modeling/bsdf/disneybrdf.cpp
//*/
//
//__forceinline__ __device__ float schlick_fresnel( const float u ) { const float m = saturate( 1.0f - u ), m2 = sqr( m ), m4 = sqr( m2 ); return m4 * m; }
//__forceinline__ __device__ void mix_spectra( const float3& a, const float3& b, const float t, float3& result ) { result = (1.0f - t) * a + t * b; }
//__forceinline__ __device__ void mix_one_with_spectra( const float3& b, const float t, float3& result ) { result = (1.0f - t) + t * b; }
//__forceinline__ __device__ void mix_spectra_with_one( const float3& a, const float t, float3& result ) { result = (1.0f - t) * a + t; }
//__forceinline__ __device__ float clearcoat_roughness( const ShadingData& shadingData ) { return mix( 0.1f, 0.001f, CLEARCOATGLOSS ); }
//__forceinline__ __device__ void DisneySpecularFresnel( const ShadingData& shadingData, const float3& o, const float3& h, float3& value )
//{
//	mix_one_with_spectra( TINT, SPECTINT, value );
//	value *= SPECULAR * 0.08f;
//	mix_spectra( value, shadingData.color, METALLIC, value );
//	const float cos_oh = fabs( dot( o, h ) );
//	mix_spectra_with_one( value, schlick_fresnel( cos_oh ), value );
//}
//__forceinline__ __device__ void DisneyClearcoatFresnel( const ShadingData& shadingData, const float3& o, const float3& h, float3& value )
//{
//	const float cos_oh = fabs( dot( o, h ) );
//	value = make_float3( mix( 0.04f, 1.0f, schlick_fresnel( cos_oh ) ) * 0.25f * CLEARCOAT );
//}
//__forceinline__ __device__ bool force_above_surface( float3& direction, const float3& normal )
//{
//	const float cos_theta = dot( direction, normal );
//	const float correction = 1.0e-4f - cos_theta;
//	if (correction <= 0) return false;
//	direction = normalize( direction + correction * normal );
//	return true;
//}
//__forceinline__ __device__ float Fr_L( float VDotN, float eio )
//{
//	if (VDotN < 0.0f) eio = 1.0f / eio, VDotN = fabs( VDotN );
//	const float SinThetaT2 = (1.0f - sqr( VDotN )) * sqr( eio );
//	if (SinThetaT2 > 1.0f) return 1.0f; // TIR
//	const float LDotN = min( sqrtf( max( 0.0f, 1.0f - SinThetaT2 ) ), 1.0f );
//	const float r1 = (VDotN - eio * LDotN) / (VDotN + eio * LDotN);
//	const float r2 = (LDotN - eio * VDotN) / (LDotN + eio * VDotN);
//	return 0.5f * (sqr( r1 ) + sqr( r2 ));
//}
//__forceinline__ __device__ bool Refract_L( const float3& wi, const float3& n, const float eta, float3& wt )
//{
//	const float cosThetaI = fabs( dot( n, wi ) );
//	const float sin2ThetaI = max( 0.0f, 1.0f - cosThetaI * cosThetaI );
//	const float sin2ThetaT = eta * eta * sin2ThetaI;
//	if (sin2ThetaT >= 1) return false; // TIR
//	const float cosThetaT = sqrtf( 1.0f - sin2ThetaT );
//	wt = eta * (wi * -1.0f) + (eta * cosThetaI - cosThetaT) * float3( n );
//	return true;
//}
//
//template <unsigned MDF>
//__forceinline__ __device__ void sample_mf( const ShadingData& shadingData, const float r0, const float r1, const float alpha_x, const float alpha_y,
//	/* const float3& gN, */ const float3& wol, /* OUT: */ float3& wil, float& pdf, float3& value )
//{
//	if (wol.z == 0) { value = make_float3( 0 ); pdf = 0; return; }
//	// compute the incoming direction by sampling the MDF
//	const float3 m = MDF == GGXMDF ? GGXMDF_sample( wol, r0, r1, alpha_x, alpha_y ) : GTR1MDF_sample( r0, r1, alpha_x, alpha_y );
//	wil = reflect( wol * -1.0f, m );
//	// force the outgoing direction to lie above the geometric surface. NOTE: does not play nice with normal interpolation; dark spots
//	// const float3 ng = World2Tangent( gN, N, T, B );
//	// if (force_above_surface( wil, ng )) m = normalize( wol + wil );
//	if (wil.z == 0) return;
//	const float cos_oh = dot( wol, m );
//	pdf = (MDF == GGXMDF ? GGXMDF_pdf( wol, m, alpha_x, alpha_y ) : GTR1MDF_pdf( wol, m, alpha_x, alpha_y )) / fabs( 4.0f * cos_oh );
//	if (pdf < 1.0e-6f) return; // skip samples with very low probability
//	const float D = MDF == GGXMDF ? GGXMDF_D( m, alpha_x, alpha_y ) : GTR1MDF_D( m, alpha_x, alpha_y );
//	const float G = MDF == GGXMDF ? GGXMDF_G( wil, wol, m, alpha_x, alpha_y ) : GTR1MDF_G( wil, wol, m, alpha_x, alpha_y );
//	if (MDF == GGXMDF) DisneySpecularFresnel( shadingData, wol, m, value ); else DisneyClearcoatFresnel( shadingData, wol, m, value );
//	value *= D * G; // postponed: / fabs( 4.0f * wol.z * wil.z );
//}
//
//template <unsigned MDF>
//__forceinline__ __device__ float evaluate_mf( const ShadingData& shadingData, const float alpha_x, const float alpha_y, const float3& wol, const float3& wil, const float3& m, float3& bsdf )
//{
//	if (wol.z == 0 || wil.z == 0) return 0;
//	// flip the incoming and outgoing vectors to be in the same hemisphere as the shading normal if needed.
//	const float cos_oh = dot( wol, m );
//	if (cos_oh == 0) return 0;
//	const float D = MDF == GGXMDF ? GGXMDF_D( m, alpha_x, alpha_y ) : GTR1MDF_D( m, alpha_x, alpha_y );
//	const float G = MDF == GGXMDF ? GGXMDF_G( wil, wol, m, alpha_x, alpha_y ) : GTR1MDF_G( wil, wol, m, alpha_x, alpha_y );
//	if (MDF == GGXMDF) DisneySpecularFresnel( shadingData, wol, m, bsdf ); else DisneyClearcoatFresnel( shadingData, wol, m, bsdf );
//	bsdf *= D * G / fabs( 4.0f * wol.z * wil.z );
//	return (MDF == GGXMDF ? GGXMDF_pdf( wol, m, alpha_x, alpha_y ) : GTR1MDF_pdf( wol, m, alpha_x, alpha_y )) / fabs( 4.0f * cos_oh );
//}
//
//__forceinline__ __device__ float evaluate_diffuse( const ShadingData& shadingData, const float3& iN, const float3& wow, const float3& wiw, const float3& m, float3& value )
//{
//	const float cos_on = dot( iN, wow );
//	const float cos_in = dot( iN, wiw );
//	const float cos_ih = dot( wiw, m );
//	const float fl = schlick_fresnel( cos_in );
//	const float fv = schlick_fresnel( cos_on );
//	float fd = 0;
//	if (SUBSURFACE != 1.0f)
//	{
//		const float fd90 = 0.5f + 2.0f * sqr( cos_ih ) * ROUGHNESS;
//		fd = mix( 1.0f, fd90, fl ) * mix( 1.0f, fd90, fv );
//	}
//	if (SUBSURFACE > 0)
//	{
//		// Based on Hanrahan-Krueger BRDF approximation of isotropic BSRDF. 1.25 is used to (roughly) preserve albedo.
//		const float fss90 = sqr( cos_ih ) * ROUGHNESS; // "flatten" retroreflection based on roughness
//		const float fss = mix( 1.0f, fss90, fl ) * mix( 1.0f, fss90, fv );
//		const float ss = 1.25f * (fss * (1.0f / (fabs( cos_on ) + fabs( cos_in )) - 0.5f) + 0.5f);
//		fd = mix( fd, ss, SUBSURFACE );
//	}
//	value = shadingData.color * fd * INVPI * (1.0f - METALLIC);
//	return fabs( cos_in ) * INVPI;
//}
//
//__forceinline__ __device__ float evaluate_sheen( const ShadingData& shadingData, const float3& wow, const float3& wiw, const float3& m, float3& value )
//{
//	// this code is mostly ported from the GLSL implementation in Disney's BRDF explorer.
//	const float3 h( normalize( wow + wiw ) );
//	const float cos_ih = dot( wiw, m );
//	const float fh = schlick_fresnel( cos_ih );
//	mix_one_with_spectra( TINT, SHEENTINT, value );
//	value *= fh * SHEEN * (1.0f - METALLIC);
//	return 1.0f / (2 * PI); // return the probability density of the sampled direction
//}
//
//__forceinline__ __device__ float3 SampleBSDF( const ShadingData& shadingData, float3 iN, const float3& N, const float3& iT, const float3& wow, const float distance,
//	const float r0, const float r1, const float r2, float3& wiw, float& pdf, bool& specular
//#ifdef __CUDACC__
//	, bool adjoint = false
//#endif
//)
//{
//	// flip interpolated normal if we arrived on the backside of a primitive
//	const float flip = (dot( wow, N ) < 0) ? -1 : 1;
//	iN *= flip;
//	// calculate tangent matrix
//	const float3 B = normalize( cross( iN, iT ) );
//	const float3 T = normalize( cross( iN, B ) );
//	// consider (rough) dielectrics
//	if (r0 < TRANSMISSION)
//	{
//		specular = true;
//		const float r3 = r0 / TRANSMISSION;
//		const float3 wol = World2Tangent( wow, iN, T, B );
//		const float eta = flip < 0 ? (1 / ETA) : ETA;
//		if (eta == 1) return make_float3( 0 );
//		const float3 beer = make_float3(
//			expf( -shadingData.transmittance.x * distance * 2.0f ),
//			expf( -shadingData.transmittance.y * distance * 2.0f ),
//			expf( -shadingData.transmittance.z * distance * 2.0f ) );
//		float alpha_x, alpha_y;
//		microfacet_alpha_from_roughness( ROUGHNESS, ANISOTROPIC, alpha_x, alpha_y );
//		const float3 m = GGXMDF_sample( wol, r1, r3, alpha_x, alpha_y );
//		const float rcp_eta = 1 / eta, cos_wom = clamp( dot( wol, m ), -1.0f, 1.0f );
//		float cos_theta_t, jacobian;
//		const float F = fresnel_reflectance( cos_wom, eta, cos_theta_t );
//		float3 wil, retVal;
//		if (r2 < F) // compute the reflected direction
//		{
//			wil = reflect( wol * -1.0f, m );
//			if (wil.z * wol.z <= 0) return make_float3( 0 );
//			evaluate_reflection( shadingData.color, wol, wil, m, alpha_x, alpha_y, F, retVal );
//			pdf = F, jacobian = reflection_jacobian( wol, m, cos_wom, alpha_x, alpha_y );
//		}
//		else // compute refracted direction
//		{
//			wil = refracted_direction( wol, m, cos_wom, cos_theta_t, eta );
//			if (wil.z * wol.z > 0) return make_float3( 0 );
//			evaluate_refraction( rcp_eta, shadingData.color, adjoint, wol, wil, m, alpha_x, alpha_y, 1 - F, retVal );
//			pdf = 1 - F, jacobian = refraction_jacobian( wol, wil, m, alpha_x, alpha_y, rcp_eta );
//		}
//		pdf *= jacobian * GGXMDF_pdf( wol, m, alpha_x, alpha_y );
//		if (pdf > 1.0e-6f) wiw = Tangent2World( wil, iN, T, B );
//		return retVal * beer;
//	}
//	// not a dielectric: normalize r0 to r3
//	const float r3 = (r0 - TRANSMISSION) / (1 - TRANSMISSION);
//	// compute component weights and cdf
//	float4 weights = make_float4( lerp( LUMINANCE, 0, METALLIC ), lerp( SHEEN, 0, METALLIC ), lerp( SPECULAR, 1, METALLIC ), CLEARCOAT * 0.25f );
//	weights *= 1.0f / (weights.x + weights.y + weights.z + weights.w);
//	const float4 cdf = make_float4( weights.x, weights.x + weights.y, weights.x + weights.y + weights.z, 0 );
//	// sample a random component
//	float probability, component_pdf;
//	float3 contrib, value = make_float3( 0 );
//	if (r3 < cdf.y)
//	{
//		const float r2 = r3 / cdf.y; // reuse r3 after normalization
//		wiw = DiffuseReflectionCosWeighted( r2, r1, iN, T, B );
//		const float3 m = normalize( wiw + wow );
//		// compute the component value and the probability density of the sampled direction.
//		if (r3 < cdf.x)
//		{
//			// Disney's diffuse
//			component_pdf = evaluate_diffuse( shadingData, iN, wow, wiw, m, value );
//			probability = DIFFWEIGHT * component_pdf, DIFFWEIGHT = 0;
//		}
//		else
//		{
//			// Disney's sheen
//			component_pdf = evaluate_sheen( shadingData, wow, wiw, m, value );
//			probability = SHEENWEIGHT * component_pdf, SHEENWEIGHT = 0;
//		}
//	}
//	else
//	{
//		const float3 wol = World2Tangent( wow, iN, T, B );
//		float3 wil;
//		if (r3 < cdf.z)
//		{
//			// Disney's specular
//			const float r2 = (r3 - cdf.y) / (cdf.z - cdf.y); // reuse r3 after normalization
//			float alpha_x, alpha_y;
//			microfacet_alpha_from_roughness( ROUGHNESS, ANISOTROPIC, alpha_x, alpha_y );
//			sample_mf<GGXMDF>( shadingData, r2, r1, alpha_x, alpha_y, wol, wil, component_pdf, value );
//			probability = SPECWEIGHT * component_pdf, SPECWEIGHT = 0;
//		}
//		else
//		{
//			// Disney's clearcoat
//			const float r2 = (r3 - cdf.z) / (1 - cdf.z); // reuse r3 after normalization
//			const float alpha = clearcoat_roughness( shadingData );
//			sample_mf<GTR1MDF>( shadingData, r2, r1, alpha, alpha, wol, wil, component_pdf, value );
//			probability = COATWEIGHT * component_pdf, COATWEIGHT = 0;
//		}
//		value *= 1.0f / fabs( 4.0f * wol.z * wil.z );
//		wiw = Tangent2World( wil, iN, T, B );
//	}
//	if (DIFFWEIGHT + SHEENWEIGHT > 0)
//	{
//		const float3 m = normalize( wiw + wow );
//		if (DIFFWEIGHT > 0) probability += DIFFWEIGHT * evaluate_diffuse( shadingData, iN, wow, wiw, m, contrib ), value += contrib;
//		if (SHEENWEIGHT > 0) probability += SHEENWEIGHT * evaluate_sheen( shadingData, wow, wiw, m, contrib ), value += contrib;
//	}
//	if (SPECWEIGHT + COATWEIGHT > 0)
//	{
//		const float3 wol = World2Tangent( wow, iN, T, B );
//		const float3 wil = World2Tangent( wiw, iN, T, B );
//		const float3 m = normalize( wol + wil );
//		if (SPECWEIGHT > 0)
//		{
//			float alpha_x, alpha_y;
//			microfacet_alpha_from_roughness( ROUGHNESS, ANISOTROPIC, alpha_x, alpha_y );
//			probability += SPECWEIGHT * evaluate_mf<GGXMDF>( shadingData, alpha_x, alpha_y, wol, wil, m, contrib );
//			value += contrib;
//		}
//		if (COATWEIGHT > 0)
//		{
//			const float alpha = clearcoat_roughness( shadingData );
//			probability += COATWEIGHT * evaluate_mf<GTR1MDF>( shadingData, alpha, alpha, wol, wil, m, contrib );
//			value += contrib;
//		}
//	}
//	if (probability > 1.0e-6f) pdf = probability; else pdf = 0;
//	return value;
//}
//
//__forceinline__ __device__ float3 EvaluateBSDF( const ShadingData& shadingData, const float3& iN, const float3& iT, const float3& wow, const float3& wiw, float& pdf )
//{
//	if (TRANSMISSION > 0.5f)
//	{
//		const float3 B = normalize( cross( iN, iT ) );
//		const float3 T = normalize( cross( iN, B ) );
//		const float3 wol = World2Tangent( wow, iN, T, B );
//		const float3 wil = World2Tangent( wiw, iN, T, B );
//		const float eta = wol.z > 0 ? ETA : (1.0f / ETA);
//		if (eta == 1) { pdf = 0; return make_float3( 0 ); }
//		float alpha_x, alpha_y, jacobian;
//		microfacet_alpha_from_roughness( ROUGHNESS, ANISOTROPIC, alpha_x, alpha_y );
//		float3 retVal, m;
//		if (wil.z * wol.z >= 0) // reflection
//		{
//			m = half_reflection_vector( wol, wil );
//			const float cos_wom = dot( wol, m );
//			const float F = fresnel_reflectance( cos_wom, 1 / eta );
//			evaluate_reflection( shadingData.color, wol, wil, m, alpha_x, alpha_y, F, retVal );
//			const float r_probability = choose_reflection_probability( 1, 1, F );
//			pdf = r_probability, jacobian = reflection_jacobian( wol, m, cos_wom, alpha_x, alpha_y );
//		}
//		else // refraction
//		{
//			m = half_refraction_vector( wol, wil, eta );
//			const float cos_wom = dot( wol, m );
//			const float F = fresnel_reflectance( cos_wom, 1 / eta );
//			evaluate_refraction( eta, shadingData.color, false /* adjoint */, wol, wil, m, alpha_x, alpha_y, 1 - F, retVal );
//			const float r_probability = choose_reflection_probability( 1, 1, F );
//			pdf = 1 - r_probability, jacobian = refraction_jacobian( wol, wil, m, alpha_x, alpha_y, eta );
//		}
//		pdf *= jacobian * GGXMDF_pdf( wol, m, alpha_x, alpha_y );
//		return retVal;
//	}
//	if (ROUGHNESS <= 0.001f)
//	{
//		// no transport via explicit connections for specular vertices
//		pdf = 0;
//		return make_float3( 0 );
//	}
//	// calculate tangent matrix
//	const float3 B = normalize( cross( iN, iT ) );
//	const float3 T = normalize( cross( iN, B ) );
//	// compute component weights
//	float4 weights = make_float4( lerp( LUMINANCE, 0, METALLIC ), lerp( SHEEN, 0, METALLIC ), lerp( SPECULAR, 1, METALLIC ), CLEARCOAT * 0.25f );
//	weights *= 1.0f / (weights.x + weights.y + weights.z + weights.w);
//	// compute pdf
//	pdf = 0;
//	float3 value = make_float3( 0 );
//	if (DIFFWEIGHT + SHEENWEIGHT > 0)
//	{
//		const float3 m = normalize( wiw + wow );
//		if (DIFFWEIGHT > 0) pdf += DIFFWEIGHT * evaluate_diffuse( shadingData, iN, wow, wiw, m, value );
//		if (SHEENWEIGHT > 0) pdf += SHEENWEIGHT * evaluate_sheen( shadingData, wow, wiw, m, value );
//	}
//	if (SPECWEIGHT + COATWEIGHT > 0)
//	{
//		const float3 wol = World2Tangent( wow, iN, T, B );
//		const float3 wil = World2Tangent( wiw, iN, T, B );
//		const float3 m = normalize( wol + wil );
//		if (SPECWEIGHT > 0)
//		{
//			float alpha_x, alpha_y;
//			microfacet_alpha_from_roughness( ROUGHNESS, ANISOTROPIC, alpha_x, alpha_y );
//			float3 contrib;
//			const float spec_pdf = evaluate_mf<GGXMDF>( shadingData, alpha_x, alpha_y, wol, wil, m, contrib );
//			if (spec_pdf > 0) pdf += SPECWEIGHT * spec_pdf, value += contrib;
//		}
//		if (COATWEIGHT > 0)
//		{
//			const float alpha = clearcoat_roughness( shadingData );
//			float3 contrib;
//			const float clearcoat_pdf = evaluate_mf<GTR1MDF>( shadingData, alpha, alpha, wol, wil, m, contrib );
//			if (clearcoat_pdf > 0) pdf += COATWEIGHT * clearcoat_pdf, value += contrib;
//		}
//	}
//	return value;
//}