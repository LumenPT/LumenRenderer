#pragma once

#include "DX12Wrapper.h"

class NullNRIWrapper : public INRIWrapper
{
public:

	NullNRIWrapper() {}
	~NullNRIWrapper() {}

	void Initialize(int a_screenWidth, int a_screenHeight) override {}


protected:

};