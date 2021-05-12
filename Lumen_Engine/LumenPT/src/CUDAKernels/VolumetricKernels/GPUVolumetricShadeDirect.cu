#include "GPUVolumetricShadingKernels.cuh"
#include <device_launch_parameters.h>
#include <sutil/vec_math.h>

using namespace WaveFront;

GPU_ONLY void VolumetricShadeDirect(
	const unsigned int a_PixelIndex,
    const uint3 a_ResolutionAndDepth,
    const WaveFront::VolumetricData* a_VolumetricDataBuffer,
    WaveFront::AtomicBuffer<WaveFront::ShadowRayData>* const a_ShadowRays,
	const AtomicBuffer<TriangleLight>* const a_Lights,
    const CDF* const a_CDF,
	float3* a_Output)
{
	const auto& intersection = a_VolumetricDataBuffer[a_PixelIndex];

	if (intersection.m_ExitIntersectionT > intersection.m_EntryIntersectionT)
	{
		const int MAX_STEPS = 1000;
		const float STEP_SIZE = 1.0f;
		const float HARDCODED_DENSITY_PER_STEP = 0.001f;
		const float VOLUME_COLOR_R = 1.0f;
		const float VOLUME_COLOR_G = 1.0f;
		const float VOLUME_COLOR_B = 1.0f;

		float distance = intersection.m_ExitIntersectionT - intersection.m_EntryIntersectionT;
		int necessarySteps = int(distance / STEP_SIZE);
		int nSteps = min(necessarySteps, MAX_STEPS);
		float accumulatedDensity = 0.0f;

		for (int i = 0; i < nSteps && accumulatedDensity < 1.0f; i++)
		{
			//Volumetric sampling code goes here
			accumulatedDensity += HARDCODED_DENSITY_PER_STEP;
		}

		float r = accumulatedDensity;
		float g = accumulatedDensity;
		float b = accumulatedDensity;

		a_Output[a_PixelIndex
			* static_cast<unsigned>(LightChannel::NUM_CHANNELS)
			+ static_cast<unsigned>(LightChannel::VOLUMETRIC)] = make_float3(r, g, b);
	}
    return;
}