#pragma once

#include "Optix/optix.h"
#include "Optix/optix_stubs.h"
#include "Optix/optix_function_table.h"

#include "MemoryBuffer.h"
#include "Shaders/CppCommon/WaveFrontDataStructs/CudaKernelParamStructs.h"
#include "../Tools/FrameSnapshot.h"
#include "CudaGLTexture.h"

#include <cstdint>

class PTServiceLocator;

struct OptixDenoiserInitParams
{
	PTServiceLocator* m_ServiceLocator;
	unsigned int m_InputWidth;
	unsigned int m_InputHeight;
};

struct OptixDenoiserDenoiseParams
{
	WaveFront::PostProcessLaunchParameters* m_PostProcessLaunchParams;
	CUdeviceptr m_ColorInput;
	CUdeviceptr m_AlbedoInput;
	CUdeviceptr m_NormalInput;
	CUdeviceptr m_Output;
};

class OptixDenoiserWrapper
{
public:

	OptixDenoiserWrapper() = default;
	~OptixDenoiserWrapper();

	void Initialize(const OptixDenoiserInitParams& a_InitParams);

	void Denoise(const OptixDenoiserDenoiseParams& a_DenoiseParams);

	void UpdateDebugTextures();

	MemoryBuffer ColorInput;
	MemoryBuffer AlbedoInput;
	MemoryBuffer NormalInput;
	MemoryBuffer ColorOutput;

	FrameSnapshot::ImageBuffer m_OptixDenoiserInputTex;
	FrameSnapshot::ImageBuffer m_OptixDenoiserAlbedoInputTex;
	FrameSnapshot::ImageBuffer m_OptixDenoiserNormalInputTex;
	FrameSnapshot::ImageBuffer m_OptixDenoiserOutputTex;

protected:

	OptixDenoiser         m_Denoiser = nullptr;
	OptixDenoiserParams   m_Params = {};

	MemoryBuffer m_state;
	MemoryBuffer m_scratch;

	OptixDenoiserInitParams m_InitParams;	

};