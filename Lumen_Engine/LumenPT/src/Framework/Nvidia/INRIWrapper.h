#pragma once

struct NRDWrapperInitParams
{
	int m_InputImageWidth = -1;
	int m_InputImageHeight = -1;
};

class INRIWrapper
{
public:

	INRIWrapper() = default;
	virtual ~INRIWrapper() = default;

	virtual void Initialize(NRDWrapperInitParams a_InitParams) = 0;


protected:
};