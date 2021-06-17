#pragma once

#include "IDLSSWrapper.h"

class NullDLSSWrapper : public IDLSSWrapper
{
public:

	NullDLSSWrapper() {}
	~NullDLSSWrapper() {}

	bool Initialize(DLSSWrapperInitParams a_InitParams) override { return false; };
	bool EvaluateDLSS(
		Microsoft::WRL::ComPtr<ID3D11Resource> a_Outputbuffer,
		Microsoft::WRL::ComPtr<ID3D11Resource> a_Inputbuffer,
		const unsigned int& a_MotionVectors,
		const unsigned int& a_DepthBuffer) override { return false; };

	std::shared_ptr<DLSSWrapperInitParams> GetDLSSParams() { return m_Params == nullptr ? std::make_shared<DLSSWrapperInitParams>() : m_Params; };
		
protected:

};