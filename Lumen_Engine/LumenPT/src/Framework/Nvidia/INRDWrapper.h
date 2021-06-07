#pragma once

class PTServiceLocator;

struct NRDWrapperInitParams
{
	int m_InputImageWidth = -1;
	int m_InputImageHeight = -1;
	PTServiceLocator* m_pServiceLocator = nullptr;
};

struct DenoiseParams
{

};

class INRDWrapper
{
public:

	INRDWrapper() = default;
	virtual ~INRDWrapper() = default;

	virtual void Initialize(NRDWrapperInitParams a_InitParams) = 0;

	virtual void Denoise(DenoiseParams a_DenoiseParams) = 0;

protected:

};