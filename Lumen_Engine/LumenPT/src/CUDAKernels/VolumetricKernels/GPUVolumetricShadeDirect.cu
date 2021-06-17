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
	unsigned int& a_Seed,
    const CDF* const a_CDF,
	float3* a_Output)
{
	const auto& intersection = a_VolumetricDataBuffer[a_PixelIndex];

	if (intersection.m_ExitIntersectionT > intersection.m_EntryIntersectionT)
	{


		//Volume ray marching settings (these need to be moved elsewhere or replaced with more sensible parameters)
		const int MAX_STEPS = 5;
		const float DENSITY_PER_METER = 0.0005f;
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
			//TODO: first sample is wasted like this, will be fixed if offset is implemented
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
			
			//Light normal at sample point dotted with light direction. Invert light dir for this (light to pixel instead of pixel to light)
			//const float cosIn = fmax(dot(pixelToLightDir, surfaceData.m_Normal), 0.f);
			const float cosOut = fmax(0.f, dot(light.normal, -pixelToLightDir));
			
			//Geometry term G(x).
			const float solidAngle = (cosOut * light.area) / (lDistance * lDistance);
			
			//BSDF is equal to material color for now.
			float sampledDensity = DENSITY_PER_METER * distanceSincePrevSample;
			const auto bssdf = make_float3(sampledDensity, sampledDensity, sampledDensity) * 0.01f;
			
			//The unshadowed contribution (contributed if no obstruction is between the light and surface) takes the BRDF,
			//geometry factor and solid angle into account. Also the light radiance.
			//The only thing missing from this is the scaling with the rest of the scene based on the reservoir PDF.
			auto unshadowedPathContribution = bssdf * solidAngle * light.radiance;
			
			//Scale by the PDF of this light to compensate for all other lights not being picked.
			unshadowedPathContribution *= (1.f / pdf); 

			ShadowRayData shadowRay(
				a_PixelIndex,
				samplePosition,
				pixelToLightDir,
				lDistance - 0.2f,
				unshadowedPathContribution,
				LightChannel::VOLUMETRIC);

			a_ShadowRays->Add(&shadowRay);

			//------------END-------------

			accumulatedDensity += sampledDensity;
		}
		//a_Output[a_PixelIndex
		//	* static_cast<unsigned>(LightChannel::NUM_CHANNELS)
		//	+ static_cast<unsigned>(LightChannel::VOLUMETRIC)] = make_float3(accumulatedDensity, accumulatedDensity, accumulatedDensity);
	}
    return;
}