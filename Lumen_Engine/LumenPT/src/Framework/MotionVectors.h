#pragma once

#include <cassert>

#include "../Shaders/CppCommon/CudaDefines.h"
#include "MemoryBuffer.h"

#include <memory>


class MotionVectors
{
public:
	MotionVectors();
	~MotionVectors();

	void Init(uint2 a_Resolution);
	
private:

	uint2 m_Resolution;
	std::unique_ptr<MemoryBuffer> m_MotionVectorBuffer;
};