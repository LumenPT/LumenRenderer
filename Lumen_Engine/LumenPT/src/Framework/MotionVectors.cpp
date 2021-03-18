#include "MotionVectors.h"

#include "../CUDAKernels/WaveFrontKernels.cuh"

MotionVectors::MotionVectors(uint2 a_Resolution) :
	m_Resolution(a_Resolution)
{
    const auto numPixels = m_Resolution.x * m_Resolution.y;
	
    const unsigned motionVectorBufferEmptySize = sizeof(MotionVectorBuffer);
    const unsigned motionVectorDataStructSize = sizeof(MotionVectorData);
	
	m_MotionVectorBuffer = std::make_unique<MemoryBuffer>(
        static_cast<size_t>(motionVectorBufferEmptySize) +
        static_cast<size_t>(numPixels) *
        static_cast<size_t>(/*m_RaysPerPixel*/1) *
        static_cast<size_t>(motionVectorDataStructSize));
	
    m_MotionVectorBuffer->Write(numPixels, 0);

    MotionVectorsGenerationData motionVectorsGenerationData;
    motionVectorsGenerationData.m_MotionVectorBuffer = nullptr;
    motionVectorsGenerationData.m_ScreenResolution = m_Resolution;
	
    GenerateMotionVectors(motionVectorsGenerationData);
}

MotionVectors::~MotionVectors()
{
}
