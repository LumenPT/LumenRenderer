#pragma once

#include <cassert>

#include "../Shaders/CppCommon/CudaDefines.h"
#include "MemoryBuffer.h"

#include <memory>



struct MotionVectorData
{
	CPU_GPU MotionVectorData()
		: m_Velocity(make_float2(0.f, 0.f))
	{}

	float2 m_Velocity;
};

struct MotionVectorBuffer
{
	CPU_GPU MotionVectorBuffer()
		: m_NumPixels(1)
		, m_MotionVectorBuffer()
	{}

	CPU_GPU INLINE unsigned int GetSize() const
	{
		return m_NumPixels;
	}

	CPU_GPU INLINE void SetMotionVectorData(
		const MotionVectorData& a_Data,
		unsigned int a_PixelIndex)
	{
		assert(a_PixelIndex < m_NumPixels);
		
		m_MotionVectorBuffer[a_PixelIndex] = a_Data;
	}

	GPU_ONLY INLINE const MotionVectorData& GetMotionVectorData(unsigned int a_PixelIndex) const
	{
		assert(a_PixelIndex < m_NumPixels);
		
		return m_MotionVectorBuffer[a_PixelIndex];

	}

	/*GPU_ONLY INLINE unsigned int GetMotionVectorDataIndex(unsigned int a_PixelIndex, const unsigned int a_RayIndex = 0) const
	{

		assert(a_PixelIndex < m_NumPixels&& a_RayIndex < m_RaysPerPixel);

		return a_PixelIndex * m_RaysPerPixel + a_RayIndex;

	}*/
	
	unsigned int m_NumPixels;
	
	MotionVectorData m_MotionVectorBuffer[];
};

class MotionVectors
{
public:
	MotionVectors(uint2 a_Resolution);
	~MotionVectors();
	
private:

	uint2 m_Resolution;
	std::unique_ptr<MemoryBuffer> m_MotionVectorBuffer;
};