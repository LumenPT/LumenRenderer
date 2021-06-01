#pragma once

#include "INRDWrapper.h"

class NullNRDWrapper : public INRDWrapper
{
public:

	NullNRDWrapper() {}
	~NullNRDWrapper() {}

	void Initialize(NRDWrapperInitParams a_InitParams) override {}


protected:

};