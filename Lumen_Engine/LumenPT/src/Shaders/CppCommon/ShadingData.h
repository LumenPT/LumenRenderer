#pragma once
//#include <cuda/cuda_runtime_api.h>
//
//#define CHAR2FLT(a,s) (((float)(((a)>>s)&255))*(1.0f/255.0f))
//
//struct ShadingData
//{
//	float3 color;			//The albedo surface color.
//	float4 transmittance;
//	float4 tint;
//	uint4 parameters;		
//
//	/* 16 uchars:   x: 0..7 = metallic, 8..15 = subsurface, 16..23 = specular, 24..31 = roughness;
//					y: 0..7 = specTint, 8..15 = anisotropic, 16..23 = sheen, 24..31 = sheenTint;
//					z: 0..7 = clearcoat, 8..15 = clearcoatGloss, 16..23 = transmission, 24..31 = dummy;
//					w: eta (32-bit float). */
//
//
//
//#define METALLIC CHAR2FLT( shadingData.parameters.x, 0 )
//#define SUBSURFACE CHAR2FLT( shadingData.parameters.x, 8 )
//#define SPECULAR CHAR2FLT( shadingData.parameters.x, 16 )
//#define ROUGHNESS (max( 0.001f, CHAR2FLT( shadingData.parameters.x, 24 ) ))
//
//#define SPECTINT CHAR2FLT( shadingData.parameters.y, 0 )
//#define ANISOTROPIC CHAR2FLT( shadingData.parameters.y, 8 )
//#define SHEEN CHAR2FLT( shadingData.parameters.y, 16 )
//#define SHEENTINT CHAR2FLT( shadingData.parameters.y, 24 )
//
//#define CLEARCOAT CHAR2FLT( shadingData.parameters.z, 0 )
//#define CLEARCOATGLOSS CHAR2FLT( shadingData.parameters.z, 8 )
//#define TRANSMISSION CHAR2FLT( shadingData.parameters.z, 16 )
//#define TINT make_float3( shadingData.tint )
//
//#define LUMINANCE shadingData.tint.w
//#define DUMMY0 CHAR2FLT( shadingData.parameters.z, 24 )
//#define ETA __uint_as_float( shadingData.parameters.w )
//
//#define CHAR_INDEX_0 0
//#define CHAR_INDEX_1 8
//#define CHAR_INDEX_2 16
//#define CHAR_INDEX_3 24
//
//#if defined(__CUDACC__)
//
//	__device__ __forceinline__ void SetColor(float3& a_Data)
//	{
//		color = a_Data;
//	}
//
//	__device__ __forceinline__ void SetTransmittance(const float3& a_Data)
//	{
//		transmittance = make_float4(a_Data);
//	}
//
//	__device__ __forceinline__ void SetTint(float4& a_Data)
//	{
//		tint = a_Data;
//	}
//
//	__device__ __forceinline__ void SetMetallic(float a_Data)
//	{
//		Overwrite(parameters.x, static_cast<unsigned int>(a_Data * 255.f), CHAR_INDEX_0);
//	}
//
//	__device__ __forceinline__ void SetSubSurface(float a_Data)
//	{
//		Overwrite(parameters.x, static_cast<unsigned int>(a_Data * 255.f), CHAR_INDEX_1);
//	}
//
//	__device__ __forceinline__ void SetSpecular(float a_Data)
//	{
//		Overwrite(parameters.x, static_cast<unsigned int>(a_Data * 255.f), CHAR_INDEX_2);
//	}
//
//	__device__ __forceinline__ void SetRoughness(float a_Data)
//	{
//		Overwrite(parameters.x, static_cast<unsigned int>(a_Data * 255.f), CHAR_INDEX_3);
//	}
//
//
//	__device__ __forceinline__ void SetSpecTint(float a_Data)
//	{
//		Overwrite(parameters.y, static_cast<unsigned int>(a_Data * 255.f), CHAR_INDEX_0);
//	}
//
//	__device__ __forceinline__ void SetAnisotropic(float a_Data)
//	{
//		Overwrite(parameters.y, static_cast<unsigned int>(a_Data * 255.f), CHAR_INDEX_1);
//	}
//
//	__device__ __forceinline__ void SetSheen(float a_Data)
//	{
//		Overwrite(parameters.y, static_cast<unsigned int>(a_Data * 255.f), CHAR_INDEX_2);
//	}
//
//	__device__ __forceinline__ void SetSheenTint(float a_Data)
//	{
//		Overwrite(parameters.y, static_cast<unsigned int>(a_Data * 255.f), CHAR_INDEX_3);
//	}
//
//
//	__device__ __forceinline__ void SetClearCoat(float a_Data)
//	{
//		Overwrite(parameters.z, static_cast<unsigned int>(a_Data * 255.f), CHAR_INDEX_0);
//	}
//
//	__device__ __forceinline__ void SetClearCoatGloss(float a_Data)
//	{
//		Overwrite(parameters.z, static_cast<unsigned int>(a_Data * 255.f), CHAR_INDEX_1);
//	}
//
//	__device__ __forceinline__ void SetTransmission(float a_Data)
//	{
//		Overwrite(parameters.z, static_cast<unsigned int>(a_Data * 255.f), CHAR_INDEX_2);
//	}
//
//	__device__ __forceinline__ void SetTint(const float3& a_Data)
//	{
//		tint = make_float4(a_Data, tint.w);
//	}
//
//	__device__ __forceinline__ void SetLuminance(float a_Data)
//	{
//		tint.w = a_Data;
//	}
//
//	__device__ __forceinline__ void SetETA(float a_Data)
//	{
//		parameters.w = __float_as_uint(a_Data);
//	}
//
//	__device__ __forceinline__ void Overwrite(unsigned int& a_Data, unsigned int a_Char, unsigned int a_Offset)
//	{
//		assert(a_Char >= 0 && a_Char <= 255);
//		a_Data &= ~(255 << a_Offset);	//Overwrite with 0. Flip to make everything 1 except the overwritten chars. 
//		a_Data |= a_Char << a_Offset;	//Write new data.
//	}
//#endif
//};