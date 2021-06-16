#pragma once
#include <cassert>
#include <vector_functions.h>
#include "../../vendor/Include/Cuda/cuda/helpers.h"

auto constexpr CHAR_INDEX_0 = 0;
auto constexpr CHAR_INDEX_1 = 8;
auto constexpr CHAR_INDEX_2 = 16;
auto constexpr CHAR_INDEX_3 = 24;

#define CHAR_TO_FLOAT(a,s) (((float)(((a)>>s)&255))*(1.0f/255.0f))

struct MaterialData
{
	float4 m_Color;			//The albedo surface color.
	float4 m_Emissive;		//Emissive modifier.
	float4 m_Transmittance;	//How much light is blocked by going through a transparent surface.
	float4 m_Tint;			//The tint of reflections and sheen.
	uint4 m_Parameters;		//All shading parameters packed as one byte.


	__device__ __host__ MaterialData() = default;

	/*
	 * Default init to a value. NOTE: This does not set the roughness to non-zero if 0 is passed.
	 * Always manually set the roughness before trying to shade.
	 */
	__device__ __host__ MaterialData(float a_Data) : m_Color(make_float4(a_Data)), m_Emissive(make_float4(a_Data)), m_Transmittance(make_float4(a_Data)), m_Tint(make_float4(a_Data)), m_Parameters(make_uint4(static_cast<unsigned>(a_Data))) {}
	
	/*
	 * Set the albedo color modifier.
	 * The w component is used for alpha.
	 */
	__device__ __host__ __forceinline__ void SetColor(const float4& a_Data)
	{
		m_Color = a_Data;
	}

	__device__ __host__ __forceinline__ void SetEmissive(const float3& a_Data)
	{
		m_Emissive = make_float4(a_Data);
	}

	__device__ __host__ __forceinline__ float3 GetEmissive() const
	{
		return make_float3(m_Emissive);
	}

	__device__ __host__ __forceinline__ float4 GetColor() const
	{
		return m_Color;
	}
	
	/*
	 * Set the transmittance factor, which determines how much light gets absorbed in Beer's law.
	 */
	__device__ __host__ __forceinline__ void SetTransmittance(const float3& a_Data)
	{
		//W component is used for refractive index.
		m_Transmittance = make_float4(a_Data, m_Transmittance.w);
	}

	__device__ __host__ __forceinline__ float3 GetTransmittance() const
	{
		return make_float3(m_Transmittance);
	}

	/*
	 * Set the tint factor used for specular reflections and sheen.
	 */
	__device__ __host__ __forceinline__ void SetTint(const float3& a_Data)
	{
		m_Tint = make_float4(a_Data, m_Tint.w);
	}

	__device__ __host__ __forceinline__ float3 GetTint() const
	{
		return make_float3(m_Tint);
	}

	/*
	 * Set how metallic this surface is.
	 */
	__device__ __host__ __forceinline__ void SetMetallic(float a_Data)
	{
		Overwrite(m_Parameters.x, static_cast<unsigned int>(a_Data * 255.f), CHAR_INDEX_0);
	}

	__device__ __host__ __forceinline__ float GetMetallic() const
	{
		return CHAR_TO_FLOAT(m_Parameters.x, CHAR_INDEX_0);
	}

	/*
	 * Set how much diffuse light scatters and leaves under the surface.
	 */
	__device__ __host__ __forceinline__ void SetSubSurface(float a_Data)
	{
		Overwrite(m_Parameters.x, static_cast<unsigned int>(a_Data * 255.f), CHAR_INDEX_1);
	}

	__device__ __host__ __forceinline__ float GetSubSurface() const
	{
		return CHAR_TO_FLOAT(m_Parameters.x, CHAR_INDEX_1);
	}

	/*
	 * Set how much of a specular reflection this surface has. Good in combination with low roughness.
	 */
	__device__ __host__ __forceinline__ void SetSpecular(float a_Data)
	{
		Overwrite(m_Parameters.x, static_cast<unsigned int>(a_Data * 255.f), CHAR_INDEX_2);
	}

	__device__ __host__ __forceinline__ float GetSpecular() const
	{
		return CHAR_TO_FLOAT(m_Parameters.x, CHAR_INDEX_2);
	}
	
	/*
	 * Set how rough the microfacets on this surface are.
	 */
	__device__ __host__ __forceinline__ void SetRoughness(float a_Data)
	{
		assert(a_Data > 0.f);
		assert(a_Data <= 1.f);
		Overwrite(m_Parameters.x, static_cast<unsigned int>(a_Data * 255.f), CHAR_INDEX_3);
	}

	__device__ __host__ __forceinline__ float GetRoughness() const
	{
		return CHAR_TO_FLOAT(m_Parameters.x, CHAR_INDEX_3);
	}

	/*
	 * Set the specular tint factor. Uses tint color.
	 */
	__device__ __host__ __forceinline__ void SetSpecTint(float a_Data)
	{
		Overwrite(m_Parameters.y, static_cast<unsigned int>(a_Data * 255.f), CHAR_INDEX_0);
	}

	__device__ __host__ __forceinline__ float GetSpecTint() const
	{
		return CHAR_TO_FLOAT(m_Parameters.y, CHAR_INDEX_0);
	}

	/*
	 * Set how anisotropic this surface is.
	 */
	__device__ __host__ __forceinline__ void SetAnisotropic(float a_Data)
	{
		Overwrite(m_Parameters.y, static_cast<unsigned int>(a_Data * 255.f), CHAR_INDEX_1);
	}

	__device__ __host__ __forceinline__ float GetAnisotropic() const
	{
		return CHAR_TO_FLOAT(m_Parameters.y, CHAR_INDEX_1);
	}

	/*
	 * Set the sheen factor of this surface.
	 */
	__device__ __host__ __forceinline__ void SetSheen(float a_Data)
	{
		Overwrite(m_Parameters.y, static_cast<unsigned int>(a_Data * 255.f), CHAR_INDEX_2);
	}

	__device__ __host__ __forceinline__ float GetSheen() const
	{
		return CHAR_TO_FLOAT(m_Parameters.y, CHAR_INDEX_2);
	}

	/*
	 * Set how much the tint color should be used for sheen.
	 */
	__device__ __host__ __forceinline__ void SetSheenTint(float a_Data)
	{
		Overwrite(m_Parameters.y, static_cast<unsigned int>(a_Data * 255.f), CHAR_INDEX_3);
	}

	__device__ __host__ __forceinline__ float GetSheenTint() const
	{
		return CHAR_TO_FLOAT(m_Parameters.y, CHAR_INDEX_3);
	}

	/*
	 * Set how much of a clearcoat this material has.
	 */
	__device__ __host__ __forceinline__ void SetClearCoat(float a_Data)
	{
		Overwrite(m_Parameters.z, static_cast<unsigned int>(a_Data * 255.f), CHAR_INDEX_0);
	}

	__device__ __host__ __forceinline__ float GetClearCoat() const
	{
		return CHAR_TO_FLOAT(m_Parameters.z, CHAR_INDEX_0);
	}

	/*
	 * Set the clearcoat gloss of this material. More gloss means more specular reflections.
	 */
	__device__ __host__ __forceinline__ void SetClearCoatGloss(float a_Data)
	{
		Overwrite(m_Parameters.z, static_cast<unsigned int>(a_Data * 255.f), CHAR_INDEX_1);
	}

	__device__ __host__ __forceinline__ float GetClearCoatGloss() const
	{
		return CHAR_TO_FLOAT(m_Parameters.z, CHAR_INDEX_1);
	}
	
	/*
	 * Set the transmission factor of this material. Fully transparent at 1.0 and fully opaque at 0.0.
	 */
	__device__ __host__ __forceinline__ void SetTransmission(float a_Data)
	{
		Overwrite(m_Parameters.z, static_cast<unsigned int>(a_Data * 255.f), CHAR_INDEX_2);
	}

	__device__ __host__ __forceinline__ float GetTransmission() const
	{
		return CHAR_TO_FLOAT(m_Parameters.z, CHAR_INDEX_2);
	}

	/*
	 * Set the luminance scalar of this material.
	 */
	__device__ __host__ __forceinline__ void SetLuminance(float a_Data)
	{
		m_Tint.w = a_Data;
	}

	__device__ __host__ __forceinline__ float GetLuminance() const
	{
		return m_Tint.w;
	}

	/*
	 * Set the index of refraction for this material.
	 */
	__device__ __host__ __forceinline__ void SetRefractiveIndex(float a_Data)
	{
		m_Transmittance.w = a_Data;
	}

	__device__ __host__ __forceinline__ float GetRefractiveIndex() const
	{
		return m_Transmittance.w;
	}

	/*
	 * Overwrite parameter data.
	 */
	__device__ __host__ __forceinline__ void Overwrite(unsigned int& a_Data, unsigned int a_Char, unsigned int a_Offset)
	{
		assert(a_Char >= 0 && a_Char <= 255);
		a_Data &= ~(255 << a_Offset);	//Overwrite with 0. Flip to make everything 1 except the overwritten chars. 
		a_Data |= a_Char << a_Offset;	//Write new data.
	}
};