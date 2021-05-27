#pragma once

#include "INRIWrapper.h"

class NullNRIWrapper : public INRIWrapper
{
public:

	NullNRIWrapper() {}
	~NullNRIWrapper() {}

	void Initialize(NRDWrapperInitParams a_InitParams) override {}


protected:

};