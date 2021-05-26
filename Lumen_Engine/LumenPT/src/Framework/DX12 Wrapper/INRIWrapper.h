#pragma once

class INRIWrapper
{
public:

	INRIWrapper() = default;
	virtual ~INRIWrapper() = default;

	virtual void Initialize(int a_screenWidth, int a_screenHeight) = 0;


protected:
};