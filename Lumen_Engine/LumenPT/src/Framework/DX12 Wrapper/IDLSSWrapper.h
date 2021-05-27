#pragma once

class IDLSSWrapper
{
public:

	IDLSSWrapper() = default;
	virtual ~IDLSSWrapper() = default;

	virtual void Initialize(int a_screenWidth, int a_screenHeight) = 0;


protected:
};