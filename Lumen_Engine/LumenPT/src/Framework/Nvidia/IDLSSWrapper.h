#pragma once
#include <memory>
#include <wrl.h>
#include <d3d11.h>

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
	virtual bool EvaluateDLSS(DLSSWrapperInitParams a_InitParams, Microsoft::WRL::ComPtr<ID3D11Resource> a_Pixelbuffer, const unsigned int& a_MotionVectors) = 0;

protected:

};