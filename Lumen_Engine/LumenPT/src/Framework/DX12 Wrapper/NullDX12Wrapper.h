#pragma once

#include "DX12Wrapper.h"

class NullDX12Wrapper : public DX12Wrapper
{
public:

	NullDX12Wrapper() {}
	~NullDX12Wrapper() {}

	void Initialize(int a_screenWidth, int a_screenHeight) override {}


protected:

};