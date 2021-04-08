#include "GPUVolumetricShadingKernels.cuh"
#include <device_launch_parameters.h>
#include <sutil/vec_math.h>

using namespace WaveFront;

GPU_ONLY void VolumetricShadeDirect(
	const unsigned int a_PixelIndex,
    const uint3 a_ResolutionAndDepth,
    const WaveFront::VolumetricData* a_VolumetricDataBuffer,
    WaveFront::AtomicBuffer<WaveFront::ShadowRayData>* const a_ShadowRays,
    const WaveFront::TriangleLight* const a_Lights,
    const unsigned int a_NumLights,
    const CDF* const a_CDF,
	float3* a_Output)
{
	if (a_VolumetricDataBuffer[a_PixelIndex].m_EntryIntersectionT > 1.0f)
	{
		a_Output[a_PixelIndex
			* static_cast<unsigned>(LightChannel::NUM_CHANNELS)
			+ static_cast<unsigned>(LightChannel::VOLUMETRIC)] = make_float3(-1.0f, -1.0f, -1.0f);
	}
    return;
}