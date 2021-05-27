#pragma once

#include "INRIWrapper.h"

class NullNRDWrapper : public INRDWrapper
{
public:

	NullNRDWrapper() {}
	~NullNRDWrapper() {}

	void Initialize(NRDWrapperInitParams a_InitParams) override {}


protected:

};