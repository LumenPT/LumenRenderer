#pragma once

#include "IDLSSWrapper.h"

class NullDLSSWrapper : public IDLSSWrapper
{
public:

	NullDLSSWrapper() {}
	~NullDLSSWrapper() {}

	bool Initialize(DLSSWrapperInitParams a_InitParams) override { return false; };
	bool EvaluateDLSS(DLSSWrapperInitParams a_InitParams, Microsoft::WRL::ComPtr<ID3D11Resource> a_Pixelbuffer, const unsigned int& a_MotionVectors) override { return false; };

protected:

};