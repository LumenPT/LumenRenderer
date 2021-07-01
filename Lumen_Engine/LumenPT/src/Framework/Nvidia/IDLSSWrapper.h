#pragma once
#include <memory>
#include <wrl.h>
#include <d3d11.h>

class PTServiceLocator;
class LumenRenderer;

/// <summary>
///  DLSS Performance/Quality setting
/// </summary>
enum class DLSSMode
{
	OFF = 0,
	MAXPERF,
	BALANCED,
	MAXQUALITY,
	ULTRAPERF,
	ULTRAQUALITY,
};

struct DLSSWrapperInitParams
{
	int m_InputImageWidth = -1;
	int m_InputImageHeight = -1;
	int m_OutputImageWidth = -1;
	int m_OutputImageHeight = -1;
	DLSSMode m_DLSSMode = DLSSMode::OFF;
	PTServiceLocator* m_pServiceLocator = nullptr;
};

struct Uint2_c {
	Uint2_c(unsigned int a_X, unsigned int a_Y) :
		m_X(a_X), m_Y(a_Y) {}
	unsigned int m_X;
	unsigned int m_Y;
};

struct DLSSRecommendedSettings
{
	DLSSRecommendedSettings() = default;
	~DLSSRecommendedSettings() = default;
	float m_Sharpness = 0.01f;
	Uint2_c m_OptimalRenderSize = { 0,0 };
	Uint2_c m_MaxDynamicRenderSize = { 0,0 };
	Uint2_c m_MinDynamicRenderSize = { 0,0 };
};

class IDLSSWrapper
{
public:

	IDLSSWrapper() = default;
	virtual ~IDLSSWrapper() = default;

	virtual bool InitializeNGX(DLSSWrapperInitParams a_InitParams) = 0;
	virtual bool InitializeDLSS(std::shared_ptr<DLSSWrapperInitParams> a_InitParams) = 0;
	virtual bool EvaluateDLSS(
		Microsoft::WRL::ComPtr<ID3D11Resource> a_Outputbuffer = nullptr,
		Microsoft::WRL::ComPtr<ID3D11Resource> a_Inputbuffer = nullptr,
		Microsoft::WRL::ComPtr<ID3D11Resource> a_DepthBuffer = nullptr,
		Microsoft::WRL::ComPtr<ID3D11Resource> a_MotionVectors = nullptr,
		Microsoft::WRL::ComPtr<ID3D11Resource> a_JitterOffset = nullptr) = 0;

	virtual std::shared_ptr<DLSSWrapperInitParams> GetDLSSParams() = 0;
	virtual std::shared_ptr<DLSSRecommendedSettings> GetRecommendedSettings(Uint2_c a_SelectedResolution, DLSSMode a_DLSSQualityVal) = 0;
	bool GetNGXInitialized() { return m_NGXInitialized; };

protected:
	bool m_NGXInitialized = false;
	std::shared_ptr<DLSSWrapperInitParams> m_Params;
	std::shared_ptr<DLSSRecommendedSettings> m_OptimalSettings = std::make_shared<DLSSRecommendedSettings>();
};