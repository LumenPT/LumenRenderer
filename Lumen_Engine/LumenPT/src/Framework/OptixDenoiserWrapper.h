#pragma once

#include "Optix/optix.h"
#include "Optix/optix_stubs.h"
#include "Optix/optix_function_table.h"

#include "MemoryBuffer.h"
#include "Shaders/CppCommon/WaveFrontDataStructs/CudaKernelParamStructs.h"

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
};

class OptixDenoiserWrapper
{
public:

	OptixDenoiserWrapper() = default;
	~OptixDenoiserWrapper();

	void Initialize(const OptixDenoiserInitParams& a_InitParams);

	void Denoise(const OptixDenoiserDenoiseParams& a_DenoiseParams);

protected:

	OptixDenoiser         m_Denoiser = nullptr;
	OptixDenoiserParams   m_Params = {};

	MemoryBuffer m_state;
	MemoryBuffer m_scratch;

	/*CUdeviceptr           m_State = 0;
	uint32_t              m_StateSize = 0;
	CUdeviceptr           m_Scratch = 0;
	uint32_t              m_ScratchSize = 0;*/

	OptixDenoiserInitParams m_InitParams;
};