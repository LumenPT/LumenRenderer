#pragma once

struct DLSSWrapperInitParams
{
	int m_InputImageWidth = -1;
	int m_InputImageHeight = -1;
	int m_OutputImageWidth = -1;
	int m_OutputImageHeight = -1;
};

class IDLSSWrapper
{
public:

	IDLSSWrapper() = default;
	virtual ~IDLSSWrapper() = default;

	virtual void Initialize(DLSSWrapperInitParams a_InitParams) = 0;


protected:
};