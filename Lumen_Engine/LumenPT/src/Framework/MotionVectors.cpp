#include "MotionVectors.h"

#include "../CUDAKernels/WaveFrontKernels.cuh"

#include <cuda_runtime.h>
#include <cuda/device_launch_parameters.h>
#include <cuda/helpers.h>
#include "../../vendor/Include/sutil/vec_math.h"
#include "../Shaders/CppCommon/WaveFrontDataStructs/MotionVectorsGenerationData.h"

#include "../Tools/SnapShotProcessing.cuh"

MotionVectors::MotionVectors()
{
	m_MotionVectorBuffer = std::make_unique<MemoryBuffer>();
}

MotionVectors::~MotionVectors()
{
}

void MotionVectors::Init(uint2 a_Resolution)
{
	m_MotionVectorDirectionsTex.m_Memory = std::make_unique<CudaGLTexture>(GL_RGB32F, a_Resolution.x,
		a_Resolution.y, 3 * sizeof(float));;
	m_MotionVectorMagnitudeTex.m_Memory = std::make_unique<CudaGLTexture>(GL_RGB32F, a_Resolution.x,
		a_Resolution.y, 3 * sizeof(float));;
	
	m_Resolution = a_Resolution;
	const auto numPixels = m_Resolution.x * m_Resolution.y;

	const unsigned motionVectorBufferEmptySize = sizeof(MotionVectorBuffer);
	const unsigned motionVectorDataStructSize = sizeof(MotionVectorData);
	size_t motionVectorBufferTotalSize =
		static_cast<size_t>(motionVectorBufferEmptySize) +
		static_cast<size_t>(numPixels) *
		static_cast<size_t>(/*m_RaysPerPixel*/1) *
		static_cast<size_t>(motionVectorDataStructSize)
		;

	m_MotionVectorBuffer->Resize(motionVectorBufferTotalSize);

	m_MotionVectorBuffer->Write(numPixels, 0);
}

void MotionVectors::Update(WaveFront::MotionVectorsGenerationData& a_MotionVectorsGenerationData)
{
	if(a_MotionVectorsGenerationData.m_ScreenResolution != m_Resolution)
	{
		Init(a_MotionVectorsGenerationData.m_ScreenResolution);
	}
	
	a_MotionVectorsGenerationData.m_MotionVectorBuffer = m_MotionVectorBuffer->GetDevicePtr<MotionVectorBuffer>();
	
	GenerateMotionVectors(a_MotionVectorsGenerationData);

	/*MotionVectorsGenerationData motionVectorsGenerationData;
	motionVectorsGenerationData.m_MotionVectorBuffer = m_MotionVectorBuffer->GetDevicePtr<MotionVectorBuffer>();
	motionVectorsGenerationData.m_ScreenResolution = m_Resolution;

	GenerateMotionVectors(motionVectorsGenerationData);*/
}

void MotionVectors::GenerateDebugTextures()
{
	assert(m_MotionVectorBuffer && "MotionVectors not initialized");
	
	SeparateMotionVectorBufferCPU(m_Resolution.x * m_Resolution.y,
		m_MotionVectorBuffer->GetDevicePtr<MotionVectorBuffer>(),
		m_MotionVectorDirectionsTex.m_Memory->GetDevicePtr<float3>(),
		m_MotionVectorMagnitudeTex.m_Memory->GetDevicePtr<float3>()
	);
}
