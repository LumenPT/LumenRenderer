#pragma once
#include "MemoryBuffer.h"

struct TriangleLight;

//TODO add CDFEntry struct in ReSTIRData.h  Add parallel CDF building if possible.

class CDF
{
public:
	/*
	 * Rebuild this CDF for the given set of lights.
	 */
	void Build(int a_NumLights, TriangleLight* a_Lights);

	/*
	 * Get the device pointer.
	 */
	void* GetDevicePtr();

private:
	MemoryBuffer m_Buffer;
};
