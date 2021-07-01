#pragma once

#include <cuda_runtime_api.h>
#include "bsdf_math.cuh"
#include "ggxmdf.cuh"
#include "frosted.cuh"
#include "../Shaders/CppCommon/MaterialStructs.h"
#include <sutil/vec_math.h>


/* disney2.h - License information:
 *
 * The below code was copied from the LightHouse2 repository on GitHub: https://github.com/jbikker/lighthouse2/blob/31c76490e01ccdcd01766ea046d66687375c04da/lib/sharedBSDFs/disney.h
   In turn, LH2 has the below copyright notice:

   This code has been adapted from AppleSeed: https://appleseedhq.net
   The AppleSeed software is released under the MIT license.
   Copyright (c) 2014-2018 Esteban Tovagliari, The appleseedhq Organization.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   // https://github.com/appleseedhq/appleseed/blob/master/src/appleseed/renderer/modeling/bsdf/disneybrdf.cpp
*/

__forceinline__ __device__ float schlick_fresnel( const float u ) { const float m = saturate( 1.0f - u ), m2 = sqr( m ), m4 = sqr( m2 ); return m4 * m; }
__forceinline__ __device__ void mix_spectra( const float3& a, const float3& b, const float t, float3& result ) { result = (1.0f - t) * a + t * b; }
__forceinline__ __device__ void mix_one_with_spectra( const float3& b, const float t, float3& result ) { result = (1.0f - t) + t * b; }
__forceinline__ __device__ void mix_spectra_with_one( const float3& a, const float t, float3& result ) { result = (1.0f - t) * a + t; }
__forceinline__ __device__ float clearcoat_roughness( const MaterialData& shadingData ) { return lerp( 0.1f, 0.001f, shadingData.GetClearCoatGloss() ); }
__forceinline__ __device__ void DisneySpecularFresnel( const MaterialData& shadingData, const float3& o, const float3& h, float3& value )
{
	mix_one_with_spectra( shadingData.GetTint(), shadingData.GetSpecTint(), value );
	value *= shadingData.GetSpecular() * 0.08f;
	mix_spectra( value, make_float3(shadingData.m_Color), shadingData.GetMetallic(), value );
	const float cos_oh = fabs( dot( o, h ) );
	mix_spectra_with_one( value, schlick_fresnel( cos_oh ), value );
}
__forceinline__ __device__ void DisneyClearcoatFresnel( const MaterialData& shadingData, const float3& o, const float3& h, float3& value )
{
	const float cos_oh = fabs( dot( o, h ) );
	value = make_float3( lerp( 0.04f, 1.0f, schlick_fresnel( cos_oh ) ) * 0.25f * shadingData.GetClearCoat() );
}
__forceinline__ __device__ bool force_above_surface( float3& direction, const float3& normal )
{
	const float cos_theta = dot( direction, normal );
	const float correction = 1.0e-4f - cos_theta;
	if (correction <= 0) return false;
	direction = normalize( direction + correction * normal );
	return true;
}
__forceinline__ __device__ float Fr_L( float VDotN, float eio )
{
	if (VDotN < 0.0f) eio = 1.0f / eio, VDotN = fabs( VDotN );
	const float SinThetaT2 = (1.0f - sqr( VDotN )) * sqr( eio );
	if (SinThetaT2 > 1.0f) return 1.0f; // TIR
	const float LDotN = fminf( sqrtf( fmaxf( 0.0f, 1.0f - SinThetaT2 ) ), 1.0f );
	const float r1 = (VDotN - eio * LDotN) / (VDotN + eio * LDotN);
	const float r2 = (LDotN - eio * VDotN) / (LDotN + eio * VDotN);
	return 0.5f * (sqr( r1 ) + sqr( r2 ));
}
__forceinline__ __device__ bool Refract_L( const float3& wi, const float3& n, const float eta, float3& wt )
{
	const float cosThetaI = fabs( dot( n, wi ) );
	const float sin2ThetaI = fmaxf( 0.0f, 1.0f - cosThetaI * cosThetaI );
	const float sin2ThetaT = eta * eta * sin2ThetaI;
	if (sin2ThetaT >= 1) return false; // TIR
	const float cosThetaT = sqrtf( 1.0f - sin2ThetaT );
	wt = eta * (wi * -1.0f) + (eta * cosThetaI - cosThetaT) * float3( n );
	return true;
}

template <unsigned MDF>
__forceinline__ __device__ void sample_mf( const MaterialData& shadingData, const float r0, const float r1, const float alpha_x, const float alpha_y,
	/* const float3& gN, */ const float3& wol, /* OUT: */ float3& wil, float& pdf, float3& value )
{
	if (wol.z == 0) { value = make_float3( 0 ); pdf = 0; return; }
	// compute the incoming direction by sampling the MDF
	const float3 m = MDF == GGXMDF ? GGXMDF_sample( wol, r0, r1, alpha_x, alpha_y ) : GTR1MDF_sample( r0, r1, alpha_x, alpha_y );
	wil = reflect( wol * -1.0f, m );
	// force the outgoing direction to lie above the geometric surface. NOTE: does not play nice with normal interpolation; dark spots
	// const float3 ng = World2Tangent( gN, N, T, B );
	// if (force_above_surface( wil, ng )) m = normalize( wol + wil );
	if (wil.z == 0) return;
	const float cos_oh = dot( wol, m );
	pdf = (MDF == GGXMDF ? GGXMDF_pdf( wol, m, alpha_x, alpha_y ) : GTR1MDF_pdf( wol, m, alpha_x, alpha_y )) / fabs( 4.0f * cos_oh );
	if (pdf < 1.0e-6f) return; // skip samples with very low probability
	const float D = MDF == GGXMDF ? GGXMDF_D( m, alpha_x, alpha_y ) : GTR1MDF_D( m, alpha_x, alpha_y );
	const float G = MDF == GGXMDF ? GGXMDF_G( wil, wol, m, alpha_x, alpha_y ) : GTR1MDF_G( wil, wol, m, alpha_x, alpha_y );
	if (MDF == GGXMDF) DisneySpecularFresnel( shadingData, wol, m, value ); else DisneyClearcoatFresnel( shadingData, wol, m, value );
	value *= D * G; // postponed: / fabs( 4.0f * wol.z * wil.z );
}

template <unsigned MDF>
__forceinline__ __device__ float evaluate_mf( const MaterialData& shadingData, const float alpha_x, const float alpha_y, const float3& wol, const float3& wil, const float3& m, float3& bsdf )
{
	if (wol.z == 0 || wil.z == 0) return 0;
	// flip the incoming and outgoing vectors to be in the same hemisphere as the shading normal if needed.
	const float cos_oh = dot( wol, m );
	if (cos_oh == 0) return 0;
	const float D = MDF == GGXMDF ? GGXMDF_D( m, alpha_x, alpha_y ) : GTR1MDF_D( m, alpha_x, alpha_y );
	const float G = MDF == GGXMDF ? GGXMDF_G( wil, wol, m, alpha_x, alpha_y ) : GTR1MDF_G( wil, wol, m, alpha_x, alpha_y );
	if (MDF == GGXMDF) DisneySpecularFresnel( shadingData, wol, m, bsdf ); else DisneyClearcoatFresnel( shadingData, wol, m, bsdf );
	bsdf *= D * G / fabs( 4.0f * wol.z * wil.z );
	return (MDF == GGXMDF ? GGXMDF_pdf( wol, m, alpha_x, alpha_y ) : GTR1MDF_pdf( wol, m, alpha_x, alpha_y )) / fabs( 4.0f * cos_oh );
}

__forceinline__ __device__ float evaluate_diffuse( const MaterialData& shadingData, const float3& iN, const float3& wow, const float3& wiw, const float3& m, float3& value )
{
	const float cos_on = dot( iN, wow );
	const float cos_in = dot( iN, wiw );
	const float cos_ih = dot( wiw, m );
	const float fl = schlick_fresnel( cos_in );
	const float fv = schlick_fresnel( cos_on );
	float fd = 0;
	if (shadingData.GetSubSurface() != 1.0f)
	{
		const float fd90 = 0.5f + 2.0f * sqr( cos_ih ) * shadingData.GetRoughness();
		const float one = 1.f;
		fd = lerp( one, fd90, fl ) * lerp( one, fd90, fv );
	}
	if (shadingData.GetSubSurface() > 0)
	{
		// Based on Hanrahan-Krueger BRDF approximation of isotropic BSRDF. 1.25 is used to (roughly) preserve albedo.
		const float fss90 = sqr( cos_ih ) * shadingData.GetRoughness(); // "flatten" retroreflection based on roughness
		const float fss = lerp( 1.0f, fss90, fl ) * lerp( 1.0f, fss90, fv );
		const float ss = 1.25f * (fss * (1.0f / (fabs( cos_on ) + fabs( cos_in )) - 0.5f) + 0.5f);
		fd = lerp( fd, ss, shadingData.GetSubSurface() );
	}
	value = make_float3(shadingData.m_Color) * fd * INVPI * (1.0f - shadingData.GetMetallic());
	return fabs( cos_in ) * INVPI;
}

__forceinline__ __device__ float evaluate_sheen( const MaterialData& shadingData, const float3& wow, const float3& wiw, const float3& m, float3& value )
{
	// this code is mostly ported from the GLSL implementation in Disney's BRDF explorer.
	const float3 h( normalize( wow + wiw ) );
	const float cos_ih = dot( wiw, m );
	const float fh = schlick_fresnel( cos_ih );
	mix_one_with_spectra( shadingData.GetTint(), shadingData.GetSheenTint(), value );
	value *= fh * shadingData.GetSheen() * (1.0f - shadingData.GetMetallic());
	return 1.0f / (2 * PI); // return the probability density of the sampled direction
}


/*
 * The below function evaluates the BSDF for a set of directions during a surface interaction.
 * Note: all directions point away from the surface.
 *
 * @ShadingData contains the surface properties such as metallic/roughness/sheen/transmission etc.
 * @iN is the interpolated normal of the surface.
 * @N is the geometric surface normal without interpolation.
 * @wow is the incoming ray direction inverted.
 * @iT is the interpolated surface tangent.
 * @distance is the distance traveled through the surface.
 *
 * @r0-r2 are uniformly generated random floats between 0 and 1 (inclusive).
 *
 * @wiw is the outgoing direction.
 * @pdf Represents the PDF of the evaluated BSDF.
 * @specular will be set to true when the calculated BSDF is specular.
 * @adjoint is set to true for very thin surfaces where the surface is left right after entering.
 *
 * @return The float3 BSDF in terms of light transport density.
 */
__forceinline__ __device__ float3 SampleBSDF( const MaterialData& shadingData, float3 iN, const float3& N, const float3& iT, const float3& wow, const float distance,
	const float r0, const float r1, const float r2, float3& wiw, float& pdf, bool& specular
#ifdef __CUDACC__
	, bool adjoint = false
#endif
)
{
	// flip interpolated normal if we arrived on the backside of a primitive
	const float flip = (dot( wow, N ) < 0) ? -1 : 1;
	iN *= flip;
	// calculate tangent matrix
	const float3 B = normalize( cross( iN, iT ) );
	const float3 T = normalize( cross( iN, B ) );
	
	// consider (rough) dielectrics
	if (r0 < shadingData.GetTransmission())
	{
		
		specular = true;
		const float r3 = r0 / shadingData.GetTransmission();
		const float3 wol = World2Tangent( wow, iN, T, B );
		const float eta = flip < 0 ? (1 / shadingData.GetRefractiveIndex()) : shadingData.GetRefractiveIndex();
		if (eta == 1) return make_float3( 0 );			//NOTE: I guess this is not possible, as the ray would just continue unobstructed.
		const float3 beer = make_float3(
			expf( -shadingData.m_Transmittance.x * distance * 2.0f ),
			expf( -shadingData.m_Transmittance.y * distance * 2.0f ),
			expf( -shadingData.m_Transmittance.z * distance * 2.0f ) );
		float alpha_x, alpha_y;
		microfacet_alpha_from_roughness( shadingData.GetRoughness(), shadingData.GetAnisotropic(), alpha_x, alpha_y );
		const float3 m = GGXMDF_sample( wol, r1, r3, alpha_x, alpha_y );
		const float rcp_eta = 1 / eta, cos_wom = clamp( dot( wol, m ), -1.0f, 1.0f );
		float cos_theta_t, jacobian;
		const float F = fresnel_reflectance( cos_wom, eta, cos_theta_t );
		float3 wil, retVal;
		if (r2 < F) // compute the reflected direction
		{
			wil = reflect( wol * -1.0f, m );
			if (wil.z * wol.z <= 0) return make_float3( 0 );
			evaluate_reflection( make_float3(shadingData.m_Color), wol, wil, m, alpha_x, alpha_y, F, retVal );
			pdf = F, jacobian = reflection_jacobian( wol, m, cos_wom, alpha_x, alpha_y );
		}
		else // compute refracted direction
		{
			wil = refracted_direction( wol, m, cos_wom, cos_theta_t, eta );
			if (wil.z * wol.z > 0) return make_float3( 0 );
			evaluate_refraction( rcp_eta, make_float3(shadingData.m_Color), adjoint, wol, wil, m, alpha_x, alpha_y, 1 - F, retVal );
			pdf = 1 - F, jacobian = refraction_jacobian( wol, wil, m, alpha_x, alpha_y, rcp_eta );
		}
		pdf *= jacobian * GGXMDF_pdf( wol, m, alpha_x, alpha_y );
		if (pdf > 1.0e-6f) wiw = Tangent2World( wil, iN, T, B );
		return retVal * beer;
	}
	// not a dielectric: normalize r0 to r3
	const float r3 = (r0 - shadingData.GetTransmission()) / (1 - shadingData.GetTransmission());
	// compute component weights and cdf
	float4 weights = make_float4( lerp( shadingData.GetLuminance(), 0.f, shadingData.GetMetallic() ), lerp( shadingData.GetSheen(), 0.f, shadingData.GetMetallic() ), lerp( shadingData.GetSpecular(), 1.f, shadingData.GetMetallic() ), shadingData.GetClearCoat() * 0.25f );
	weights *= 1.0f / (weights.x + weights.y + weights.z + weights.w);
	const float4 cdf = make_float4( weights.x, weights.x + weights.y, weights.x + weights.y + weights.z, 0 );
	// sample a random component
	float probability, component_pdf;
	float3 contrib, value = make_float3( 0 );
	if (r3 < cdf.y)
	{
		const float r2 = r3 / cdf.y; // reuse r3 after normalization
		wiw = DiffuseReflectionCosWeighted( r2, r1, iN, T, B );
		const float3 m = normalize( wiw + wow );
		// compute the component value and the probability density of the sampled direction.
		if (r3 < cdf.x)
		{
			// Disney's diffuse
			component_pdf = evaluate_diffuse( shadingData, iN, wow, wiw, m, value );
			probability = DIFFWEIGHT * component_pdf, DIFFWEIGHT = 0;
		}
		else
		{
			// Disney's sheen
			component_pdf = evaluate_sheen( shadingData, wow, wiw, m, value );
			probability = SHEENWEIGHT * component_pdf, SHEENWEIGHT = 0;
		}
	}
	else
	{
		const float3 wol = World2Tangent( wow, iN, T, B );
		float3 wil;
		if (r3 < cdf.z)
		{
			// Disney's specular
			const float r2 = (r3 - cdf.y) / (cdf.z - cdf.y); // reuse r3 after normalization
			float alpha_x, alpha_y;
			microfacet_alpha_from_roughness( shadingData.GetRoughness(), shadingData.GetAnisotropic(), alpha_x, alpha_y );
			sample_mf<GGXMDF>( shadingData, r2, r1, alpha_x, alpha_y, wol, wil, component_pdf, value );
			probability = SPECWEIGHT * component_pdf, SPECWEIGHT = 0;
		}
		else
		{
			// Disney's clearcoat
			const float r2 = (r3 - cdf.z) / (1 - cdf.z); // reuse r3 after normalization
			const float alpha = clearcoat_roughness( shadingData );
			sample_mf<GTR1MDF>( shadingData, r2, r1, alpha, alpha, wol, wil, component_pdf, value );
			probability = COATWEIGHT * component_pdf, COATWEIGHT = 0;
		}
		value *= 1.0f / fabs( 4.0f * wol.z * wil.z );
		wiw = Tangent2World( wil, iN, T, B );
	}
	if (DIFFWEIGHT + SHEENWEIGHT > 0)
	{
		const float3 m = normalize( wiw + wow );
		if (DIFFWEIGHT > 0) probability += DIFFWEIGHT * evaluate_diffuse( shadingData, iN, wow, wiw, m, contrib ), value += contrib;
		if (SHEENWEIGHT > 0) probability += SHEENWEIGHT * evaluate_sheen( shadingData, wow, wiw, m, contrib ), value += contrib;
	}
	if (SPECWEIGHT + COATWEIGHT > 0)
	{
		const float3 wol = World2Tangent( wow, iN, T, B );
		const float3 wil = World2Tangent( wiw, iN, T, B );
		const float3 m = normalize( wol + wil );
		if (SPECWEIGHT > 0)
		{
			float alpha_x, alpha_y;
			microfacet_alpha_from_roughness( shadingData.GetRoughness(), shadingData.GetAnisotropic(), alpha_x, alpha_y );
			probability += SPECWEIGHT * evaluate_mf<GGXMDF>( shadingData, alpha_x, alpha_y, wol, wil, m, contrib );
			value += contrib;
		}
		if (COATWEIGHT > 0)
		{
			const float alpha = clearcoat_roughness( shadingData );
			probability += COATWEIGHT * evaluate_mf<GTR1MDF>( shadingData, alpha, alpha, wol, wil, m, contrib );
			value += contrib;
		}
	}
	if (probability > 1.0e-6f) pdf = probability; else pdf = 0;
	return value;
}

/*
 * The below function returns the BSDF for a surface interactions for an incoming ray.
 * Note: all directions point away from the surface.
 *
 * @shadingData contains the surface and material data.
 * @iN The surface normal (interpolated).
 * @iT The geometry surface tangent (interpolated).
 * @wow The inverse incoming ray direction (surface to ray origin).
 *
 * @wiw The generated outgoing direction.
 * @pdf The generated PDF for the sampled direction.
 *
 * @return The float3 BSDF in terms of light transport density.
 */
__forceinline__ __device__ float3 EvaluateBSDF( const MaterialData& shadingData, const float3& iN, const float3& iT, const float3& wow, const float3& wiw, float& pdf)
{
	float3 specTransBSDF = { 0.f };
	float specTransPDF = 0.f;
	const float transmission = shadingData.GetTransmission();
	
	if (transmission > 0.f)
	{
		const float3 B = normalize( cross( iN, iT ) );
		const float3 T = normalize( cross( iN, B ) );
		const float3 wol = World2Tangent( wow, iN, T, B );
		const float3 wil = World2Tangent( wiw, iN, T, B );
		const float eta = wol.z > 0 ? shadingData.GetRefractiveIndex() : (1.0f / shadingData.GetRefractiveIndex());
		if (eta == 1) { pdf = 0; return make_float3( 0 ); }
		float alpha_x, alpha_y, jacobian;
		microfacet_alpha_from_roughness( shadingData.GetRoughness(), shadingData.GetAnisotropic(), alpha_x, alpha_y );
		float3 m;
		if (wil.z * wol.z >= 0) // reflection
		{
			m = half_reflection_vector( wol, wil );
			const float cos_wom = dot( wol, m );
			const float F = fresnel_reflectance( cos_wom, 1 / eta );
			evaluate_reflection( make_float3(shadingData.m_Color), wol, wil, m, alpha_x, alpha_y, F, specTransBSDF);
			const float r_probability = choose_reflection_probability( 1, 1, F );
			specTransPDF = r_probability, jacobian = reflection_jacobian( wol, m, cos_wom, alpha_x, alpha_y );
		}
		else // refraction
		{
			m = half_refraction_vector( wol, wil, eta );
			const float cos_wom = dot( wol, m );
			const float F = fresnel_reflectance( cos_wom, 1 / eta );
			evaluate_refraction( eta, make_float3(shadingData.m_Color), false /* adjoint */, wol, wil, m, alpha_x, alpha_y, 1 - F, specTransBSDF);
			const float r_probability = choose_reflection_probability( 1, 1, F );
			specTransPDF = 1 - r_probability, jacobian = refraction_jacobian( wol, wil, m, alpha_x, alpha_y, eta );
		}
		specTransPDF *= jacobian * GGXMDF_pdf( wol, m, alpha_x, alpha_y );
		//return retVal; Edited out by Jan because I will mix all things together.
	}
	if (shadingData.GetRoughness() <= 0.001f)
	{
		// no transport via explicit connections for specular vertices
		pdf = specTransPDF;
		return specTransBSDF;
	}
	// calculate tangent matrix
	const float3 B = normalize( cross( iN, iT ) );
	const float3 T = normalize( cross( iN, B ) );
	// compute component weights
	float4 weights = make_float4( lerp( shadingData.GetLuminance(), 0.f, shadingData.GetMetallic() ), lerp( shadingData.GetSheen(), 0.f, shadingData.GetMetallic() ), lerp( shadingData.GetSpecular(), 1.f, shadingData.GetMetallic() ), shadingData.GetClearCoat() * 0.25f );
	weights *= 1.0f / (weights.x + weights.y + weights.z + weights.w);
	// compute pdf
	pdf = 0;
	float3 value = make_float3( 0 );
	if (DIFFWEIGHT + SHEENWEIGHT > 0)
	{
		const float3 m = normalize( wiw + wow );
		if (DIFFWEIGHT > 0) pdf += DIFFWEIGHT * evaluate_diffuse( shadingData, iN, wow, wiw, m, value );
		if (SHEENWEIGHT > 0) pdf += SHEENWEIGHT * evaluate_sheen( shadingData, wow, wiw, m, value );
	}
	if (SPECWEIGHT + COATWEIGHT > 0)
	{
		const float3 wol = World2Tangent( wow, iN, T, B );
		const float3 wil = World2Tangent( wiw, iN, T, B );
		const float3 m = normalize( wol + wil );
		if (SPECWEIGHT > 0)
		{
			float alpha_x, alpha_y;
			microfacet_alpha_from_roughness( shadingData.GetRoughness(), shadingData.GetAnisotropic(), alpha_x, alpha_y );
			float3 contrib;
			const float spec_pdf = evaluate_mf<GGXMDF>( shadingData, alpha_x, alpha_y, wol, wil, m, contrib );
			if (spec_pdf > 0) pdf += SPECWEIGHT * spec_pdf, value += contrib;
		}
		if (COATWEIGHT > 0)
		{
			const float alpha = clearcoat_roughness( shadingData );
			float3 contrib;
			const float clearcoat_pdf = evaluate_mf<GTR1MDF>( shadingData, alpha, alpha, wol, wil, m, contrib );
			if (clearcoat_pdf > 0) pdf += COATWEIGHT * clearcoat_pdf, value += contrib;
		}
	}

	//Edit by Jan: Mix between transmission and other lobes.
	pdf = (pdf * (1.f - transmission));
	pdf += (specTransPDF * transmission);
	return (specTransBSDF * transmission) + (value * (1.f - transmission));
}