#include "GPUVolumetricShadingKernels.cuh"
#include <device_launch_parameters.h>
#include <sutil/vec_math.h>
#include "../../Shaders/CppCommon/Half4.h"

using namespace WaveFront;

GPU_ONLY void VolumetricShadeDirect(
	PixelIndex a_PixelIndex,
    const uint3 a_ResolutionAndDepth,
    const WaveFront::VolumetricData* a_VolumetricDataBuffer,
    WaveFront::AtomicBuffer<WaveFront::ShadowRayData>* const a_ShadowRays,
	const AtomicBuffer<TriangleLight>* const a_Lights,
	unsigned int& a_Seed,
    const CDF* const a_CDF,
	cudaSurfaceObject_t a_Output)
{

	const unsigned int pixelDataIndex = PIXEL_DATA_INDEX(a_PixelIndex.m_X, a_PixelIndex.m_Y, a_ResolutionAndDepth.x);
	const auto& intersection = a_VolumetricDataBuffer[pixelDataIndex];
	

	if (intersection.m_ExitIntersectionT > intersection.m_EntryIntersectionT)
	{
		//Volume ray marching settings (these need to be moved elsewhere or replaced with more sensible parameters)
		const int MAX_STEPS = 5;
		const float DENSITY_PER_METER = intersection.m_Density;
		const float VOLUME_COLOR_R = 1.0f;
		const float VOLUME_COLOR_G = 1.0f;
		const float VOLUME_COLOR_B = 1.0f;

		//Sample volume
		float distance = intersection.m_ExitIntersectionT - intersection.m_EntryIntersectionT;
		float accumulatedDensity = 0.0f;
		//Calculate appropriate step size
		float stepSize = distance / MAX_STEPS;
		float3 prevSamplePosition = intersection.m_PositionEntry;
		float offset = RandomFloat(a_Seed) * stepSize;	//This is used to offset each ray into the screen by a small amount, sampling different parts of the volume

		for (int i = 0; i < MAX_STEPS && accumulatedDensity < 1.0f && (float)i * stepSize < distance; i++)
		{
			float sampleT = (float)i * stepSize + offset;
			float3 samplePosition = intersection.m_PositionEntry + intersection.m_IncomingRayDirection * sampleT;
			float distanceSincePrevSample = length(samplePosition - prevSamplePosition);
			prevSamplePosition = samplePosition;

			//Pick a light from the CDF
			unsigned index;
			float pdf;
			a_CDF->Get(RandomFloat(a_Seed), index, pdf);

			auto& light = *a_Lights->GetData(index);

			//Pick random point on light
			const float u = RandomFloat(a_Seed);
			const float v = RandomFloat(a_Seed) * (1.f - u);
			float3 arm1 = light.p1 - light.p0;
			float3 arm2 = light.p2 - light.p0;
			float3 lightCenter = light.p0 + (arm1 * u) + (arm2 * v);

			//Direction from pixel to light
			float3 pixelToLightDir = lightCenter - samplePosition;
			//Light distance from pixel
			const float lDistance = length(pixelToLightDir);
			//Normalize
			pixelToLightDir /= lDistance;
			
			float sampledDensity = DENSITY_PER_METER * distanceSincePrevSample;

			auto volumeColor = make_float3(VOLUME_COLOR_R, VOLUME_COLOR_G, VOLUME_COLOR_B) * 0.01f;

			ShadowRayData shadowRay(
				a_PixelIndex,
				samplePosition,
				pixelToLightDir,
				lDistance - 0.2f,
				volumeColor,
				LightChannel::VOLUMETRIC);

			a_ShadowRays->Add(&shadowRay);

			accumulatedDensity += sampledDensity;
		}
		//printf("Density: %d", accumulatedDensity);
		half4Ushort4 color{ make_float4(0.f, 0.f, 0.f, accumulatedDensity) };
		surf2Dwrite<ushort4>(
			color.m_Ushort4,
			a_Output,
			a_PixelIndex.m_X * sizeof(ushort4),
			a_PixelIndex.m_Y,
			cudaBoundaryModeTrap);
		//surf2DLayeredwrite<float4>(
		//	make_float4(0.f, 0.f, 0.f, accumulatedDensity),
		//	a_Output,
		//	a_PixelIndex.m_X * sizeof(float4),
		//	a_PixelIndex.m_Y,
		//	static_cast<unsigned>(LightChannel::VOLUMETRIC),
		//	cudaBoundaryModeTrap);
	}
    return;
}