#pragma once

#include "Optix/optix.h"
#include "Optix/optix_stubs.h"
#include "Optix/optix_function_table.h"

#include "MemoryBuffer.h"
#include "Shaders/CppCommon/WaveFrontDataStructs/CudaKernelParamStructs.h"
#include "../Tools/FrameSnapshot.h"
#include "CudaGLTexture.h"

//#include "../CUDAKernels/WaveFrontKernels.cuh"

#include <cstdint>

class PTServiceLocator;

namespace WaveFront
{
	struct OptixDenoiserLaunchParameters;
}

struct OptixDenoiserInitParams
{
	PTServiceLocator* m_ServiceLocator;
	unsigned int m_InputWidth = -1;
	unsigned int m_InputHeight = -1;
	bool m_UseAlbedo;
	bool m_UseNormal;
	bool m_UseTemporalData;

	friend bool operator==(const OptixDenoiserInitParams& a_Left, const OptixDenoiserInitParams& a_Right)
	{
		return (
			a_Left.m_InputWidth == a_Right.m_InputWidth &&
			a_Left.m_InputHeight == a_Right.m_InputHeight &&
			a_Left.m_UseAlbedo == a_Right.m_UseAlbedo &&
			a_Left.m_UseNormal == a_Right.m_UseNormal &&
			a_Left.m_UseTemporalData == a_Right.m_UseTemporalData
			);
	}

	friend bool operator!=(const OptixDenoiserInitParams& a_Left, const OptixDenoiserInitParams& a_Right)
	{
		return !(a_Left == a_Right);
	}
};

struct OptixDenoiserDenoiseParams
{
	OptixDenoiserInitParams m_InitParams;
	WaveFront::PostProcessLaunchParameters* m_PostProcessLaunchParams;
	WaveFront::OptixDenoiserLaunchParameters* m_OptixDenoiserLaunchParams;
	CUdeviceptr m_ColorInput;
	CUdeviceptr m_AlbedoInput;
	CUdeviceptr m_NormalInput;
	CUdeviceptr m_FlowInput;
	CUdeviceptr m_PrevColorOutput;
	CUdeviceptr m_ColorOutput;
	bool m_BlendOutput;
	unsigned int m_BlendCount;
};

class OptixDenoiserWrapper
{
public:

	OptixDenoiserWrapper() = default;
	~OptixDenoiserWrapper();

	void Initialize(const OptixDenoiserInitParams& a_InitParams);

	void Denoise(OptixDenoiserDenoiseParams& a_DenoiseParams);

	void UpdateDebugTextures();

	MemoryBuffer ColorInput;
	MemoryBuffer AlbedoInput;
	MemoryBuffer NormalInput;
	MemoryBuffer FlowInput;

	MemoryBuffer BlendOutput;

	MemoryBuffer& GetColorOutput() { return ColorOutput[m_currentColorOutputIndex]; }
	MemoryBuffer& GetPrevColorOutput() { return ColorOutput[(m_currentColorOutputIndex + 1) % ms_colorOutputNum]; }

	FrameSnapshot::ImageBuffer m_OptixDenoiserInputTex;
	FrameSnapshot::ImageBuffer m_OptixDenoiserAlbedoInputTex;
	FrameSnapshot::ImageBuffer m_OptixDenoiserNormalInputTex;
	FrameSnapshot::ImageBuffer m_OptixDenoiserOutputTex;

protected:

	static const size_t ms_colorOutputNum = 2;
	std::array<MemoryBuffer, OptixDenoiserWrapper::ms_colorOutputNum> ColorOutput;
	size_t m_currentColorOutputIndex = 0;

	OptixDenoiser         m_Denoiser = nullptr;
	OptixDenoiserParams   m_Params = {};

	MemoryBuffer m_state;
	MemoryBuffer m_scratch;

	OptixDenoiserInitParams m_InitParams;	

};