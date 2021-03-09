#pragma once
#include "MemoryBuffer.h"

namespace WaveFront
{
	struct TriangleLight;
}

//TODO add CDFEntry struct in ReSTIRData.h  Add parallel CDF building if possible.

class CDF
{
public:
	/*
	 * Rebuild this CDF for the given set of lights.
	 */
	void Build(int a_NumLights, WaveFront::TriangleLight* a_Lights);

	/*
	 * Fill a light bag buffer with lights from this CDF.
	 */
	void* FillLightBags(void* a_LightBags, int a_NumLights);

	/*
	 * Get the device pointer.
	 */
	void* GetDevicePtr();

private:
	MemoryBuffer m_Buffer;
};
