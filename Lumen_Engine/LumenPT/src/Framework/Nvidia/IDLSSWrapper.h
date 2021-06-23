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

struct uint2_c {
	unsigned int m_X;
	unsigned int m_Y;
};

struct DLSSRecommendedSettings
{
	float m_Sharpness = 0.01f;
	uint2_c m_OptimalRenderSize;
	uint2_c m_MaxDynamicRenderSize;
	uint2_c m_MinDynamicRenderSize;
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
		Microsoft::WRL::ComPtr<ID3D11Resource> a_DepthBuffer = nullptr,
		Microsoft::WRL::ComPtr<ID3D11Resource> a_MotionVectors = nullptr,
		Microsoft::WRL::ComPtr<ID3D11Resource> a_JitterOffset = nullptr) = 0;

	virtual std::shared_ptr<DLSSWrapperInitParams> GetDLSSParams() = 0;
	virtual std::shared_ptr<DLSSRecommendedSettings> GetRecommendedSettings() = 0;
	bool GetNGXInitialized() { return m_ngxInitialized; };

protected:
	bool m_ngxInitialized = false;
	std::shared_ptr<DLSSWrapperInitParams> m_Params;
	std::shared_ptr<DLSSRecommendedSettings> m_OptimalSettings;
};