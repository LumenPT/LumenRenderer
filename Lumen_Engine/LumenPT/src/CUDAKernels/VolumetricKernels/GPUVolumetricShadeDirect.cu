#include "GPUVolumetricShadingKernels.cuh"
#include <device_launch_parameters.h>
#include <sutil/vec_math.h>

using namespace WaveFront;

GPU_ONLY void VolumetricShadeDirect(
	PixelIndex a_PixelIndex,
    const uint3 a_ResolutionAndDepth,
    const WaveFront::VolumetricData* a_VolumetricDataBuffer,
    WaveFront::AtomicBuffer<WaveFront::ShadowRayData>* const a_ShadowRays,
	const AtomicBuffer<TriangleLight>* const a_Lights,
    const CDF* const a_CDF,
	cudaSurfaceObject_t a_Output)
{

	const unsigned int pixelDataIndex = PIXEL_DATA_INDEX(a_PixelIndex.m_X, a_PixelIndex.m_Y, a_ResolutionAndDepth.x);
	const auto& intersection = a_VolumetricDataBuffer[pixelDataIndex];
	

	if (intersection.m_ExitIntersectionT > intersection.m_EntryIntersectionT)
	{
		const int MAX_STEPS = 1000;
		const float STEP_SIZE = 1.0f;
		const float HARDCODED_DENSITY_PER_STEP = 0.01f;
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

		surf2DLayeredwrite<float4>(
			make_float4(r, g, b, 1.f),
			a_Output,
			a_PixelIndex.m_X * sizeof(float4),
			a_PixelIndex.m_Y,
			static_cast<unsigned>(LightChannel::VOLUMETRIC),
			cudaBoundaryModeTrap);
	}
    return;
}