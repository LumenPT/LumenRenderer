#pragma once

class PTServiceLocator;

struct DLSSWrapperInitParams
{
	int m_InputImageWidth = -1;
	int m_InputImageHeight = -1;
	int m_OutputImageWidth = -1;
	int m_OutputImageHeight = -1;
	PTServiceLocator* m_pServiceLocator = nullptr;
};



class IDLSSWrapper
{
public:

	IDLSSWrapper() = default;
	virtual ~IDLSSWrapper() = default;

	virtual bool Initialize(DLSSWrapperInitParams a_InitParams) = 0;
	virtual bool EvaluateDLSS() = 0;

protected:

};