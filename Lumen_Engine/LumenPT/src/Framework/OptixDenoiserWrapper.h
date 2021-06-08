#pragma once

#include "Optix/optix.h"
#include "Optix/optix_stubs.h"

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

	CUdeviceptr           m_Scratch = 0;
	uint32_t              m_ScratchSize = 0;
	CUdeviceptr           m_State = 0;
	uint32_t              m_StateSize = 0;

	OptixDenoiserInitParams m_InitParams;
};