#pragma once

class DX12Wrapper
{
public:

	DX12Wrapper() = default;
	virtual ~DX12Wrapper() = default;

	virtual void Initialize(int a_screenWidth, int a_screenHeight) = 0;


protected:
};