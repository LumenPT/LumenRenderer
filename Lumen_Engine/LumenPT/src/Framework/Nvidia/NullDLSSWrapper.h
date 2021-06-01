#pragma once

#include "IDLSSWrapper.h"

class NullDLSSWrapper : public IDLSSWrapper
{
public:

	NullDLSSWrapper() {}
	~NullDLSSWrapper() {}

	bool Initialize(DLSSWrapperInitParams a_InitParams) override {}


protected:

};