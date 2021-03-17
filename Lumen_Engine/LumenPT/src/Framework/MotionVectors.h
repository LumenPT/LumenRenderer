#pragma once

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