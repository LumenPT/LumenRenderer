#pragma once

#include <cassert>

#include "../Shaders/CppCommon/CudaDefines.h"
#include "MemoryBuffer.h"
#include "../Tools/FrameSnapshot.h"
#include "CudaGLTexture.h"

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

	MemoryBuffer* GetMotionVectorBuffer() { return m_MotionVectorBuffer.get(); };

	void GenerateDebugTextures();
	
	GLuint GetMotionVectorDirectionsTex() { return m_MotionVectorDirectionsTex.m_Memory->GetTexture(); };
	GLuint GetMotionVectorMagnitudeTex() { return m_MotionVectorMagnitudeTex.m_Memory->GetTexture(); };
	
private:

	FrameSnapshot::ImageBuffer m_MotionVectorDirectionsTex;
	FrameSnapshot::ImageBuffer m_MotionVectorMagnitudeTex;
	
	uint2 m_Resolution;
	std::unique_ptr<MemoryBuffer> m_MotionVectorBuffer;
};