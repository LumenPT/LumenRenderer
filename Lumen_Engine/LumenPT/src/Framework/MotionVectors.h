#pragma once

#include <cassert>

#include "../Shaders/CppCommon/CudaDefines.h"
#include "MemoryBuffer.h"

#include <memory>

namespace WaveFront
{
	struct MotionVectorsGenerationData;
}

class MotionVectors
{
public:
	MotionVectors();
	~MotionVectors();

	void Init(uint2 a_Resolution);
	void Update(WaveFront::MotionVectorsGenerationData& a_MotionVectorsGenerationData);
	
private:

	uint2 m_Resolution;
	std::unique_ptr<MemoryBuffer> m_MotionVectorBuffer;
};