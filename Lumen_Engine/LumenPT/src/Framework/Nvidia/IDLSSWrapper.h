#pragma once
#include <memory>
#include <wrl.h>
#include <d3d11.h>

class PTServiceLocator;
class LumenRenderer;

struct DLSSWrapperInitParams
{
	enum class DLSSMode 
	{
		OFF = 0,
		MAXPERF,
		BALANCED,
		MAXQUALITY,
		ULTRAPERF,
		ULTRAQUALITY,
	};
	int m_InputImageWidth = -1;
	int m_InputImageHeight = -1;
	int m_OutputImageWidth = -1;
	int m_OutputImageHeight = -1;
	DLSSMode m_DLSSMode = DLSSMode::OFF;
	PTServiceLocator* m_pServiceLocator = nullptr;
};



class IDLSSWrapper
{
public:

	IDLSSWrapper() = default;
	virtual ~IDLSSWrapper() = default;

	virtual bool Initialize(DLSSWrapperInitParams a_InitParams) = 0;
	virtual bool EvaluateDLSS(
		Microsoft::WRL::ComPtr<ID3D11Resource> a_Outputbuffer = nullptr, 
		Microsoft::WRL::ComPtr<ID3D11Resource> a_Inputbuffer = nullptr, 
		const unsigned int& a_MotionVectors = 0, 
		const unsigned int& a_DepthBuffer = 0) = 0;

	virtual std::shared_ptr<DLSSWrapperInitParams> GetDLSSParams() = 0;

protected:
	std::shared_ptr<DLSSWrapperInitParams> m_Params;
};