#pragma once

#include "IDLSSWrapper.h"

class NullDLSSWrapper : public IDLSSWrapper
{
public:

	NullDLSSWrapper() {}
	~NullDLSSWrapper() {}
	
	bool InitializeNGX(DLSSWrapperInitParams a_InitParams) override { return false; };
	bool InitializeDLSS(std::shared_ptr<DLSSWrapperInitParams> a_InitParams) override { return false; };

	virtual bool EvaluateDLSS(
		Microsoft::WRL::ComPtr<ID3D11Resource> a_Outputbuffer = nullptr,
		Microsoft::WRL::ComPtr<ID3D11Resource> a_Inputbuffer = nullptr,
		Microsoft::WRL::ComPtr<ID3D11Resource> a_DepthBuffer = nullptr,
		Microsoft::WRL::ComPtr<ID3D11Resource> a_MotionVectors = nullptr,
		Microsoft::WRL::ComPtr<ID3D11Resource> a_JitterOffset = nullptr) override
	{
		return false;
	};

	std::shared_ptr<DLSSWrapperInitParams> GetDLSSParams() { return m_Params == nullptr ? std::make_shared<DLSSWrapperInitParams>() : m_Params; };
	std::shared_ptr<DLSSRecommendedSettings> GetRecommendedSettings(Uint2_c a_SelectedResolution, DLSSMode a_DLSSQualityVal) override { return m_OptimalSettings; };

protected:

};