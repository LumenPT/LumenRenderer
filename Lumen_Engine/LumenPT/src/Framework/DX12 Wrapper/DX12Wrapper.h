#pragma once

class DX12Wrapper
{
public:

	DX12Wrapper() = default;
	virtual ~DX12Wrapper() = default;

	virtual void Initialize() = 0;


protected:
};