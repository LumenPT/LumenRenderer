#include "Timer.h"
#ifdef WAVEFRONT
#include "ReSTIR.h"

#include <cassert>
#include <cuda_runtime.h>

#include "../CUDAKernels/RandomUtilities.cuh"
#include "../Shaders/CppCommon/WaveFrontDataStructs/AtomicBuffer.h"
#include "OptixWrapper.h"

CPU_ONLY void ReSTIR::Initialize(const ReSTIRSettings& a_Settings)
{
	m_Settings = a_Settings;

	//Ensure correct configuration.
	assert(m_Settings.width != 0 && m_Settings.height != 0 && "ReSTIR requires screen dimensions to be non-zero positive values.");
	assert(m_Settings.numLightsPerBag > 0 && "Num lights per bag needs to be at least 1.");
	assert(m_Settings.numPrimarySamples > 0 && "Num primary samples needs to be at least 1.");
	assert(m_Settings.numReservoirsPerPixel > 0 && "Num reservoirs per pixel needs to be at least 1.");
	assert(m_Settings.numSpatialIterations > 0 && "The amount of spatial iterations needs to be at least 1.");
	assert(m_Settings.numSpatialIterations % 2 == 0 && "The amount of spatial iterations needs to be an even number (so that final output ends up in the right buffer).");
	assert(m_Settings.numSpatialSamples > 0 && "The amount of spatial samples needs to be at least 1.");
	assert(m_Settings.pixelGridSize > 0 && "The pixel grid size needs to be at least 1 by 1 pixels.");
	assert(m_Settings.spatialSampleRadius > 0 && "The spatial sample radius needs to be at least 1 pixel.");

	//Initialize the buffers required.

	//Shadow rays.
	{
		//At most one shadow ray per reservoir. Always resize even when big enough already, because this is initialization so it should not happen often.
		const size_t size = static_cast<size_t>(m_Settings.width) * static_cast<size_t>(m_Settings.height) * m_Settings.numReservoirsPerPixel;
		WaveFront::CreateAtomicBuffer<RestirShadowRay>(&m_ShadowRays, size);
	}

	//Reservoirs
	{
		//Reserve enough memory for both the front and back buffer to contain all reservoirs.
		const size_t numReservoirs = static_cast<size_t>(m_Settings.width) * static_cast<size_t>(m_Settings.height) * m_Settings.numReservoirsPerPixel;
		const size_t size = numReservoirs * sizeof(Reservoir);

		//Initialize all three reservoir buffers.
		for(int reservoir = 0; reservoir < 3; ++reservoir)
		{
			m_Reservoirs[reservoir].Resize(size);
			ResetReservoirs(numReservoirs, static_cast<Reservoir*>(m_Reservoirs[reservoir].GetDevicePtr()));
		}
	}

	//Light bag generation
	{
		const size_t size = sizeof(LightBagEntry) * m_Settings.numLightsPerBag * m_Settings.numLightBags;
		if (m_LightBags.GetSize() < size)
		{
			m_LightBags.Resize(size);
		}
	}

	//Wait for CUDA to finish executing.
	cudaDeviceSynchronize();
}

CPU_ONLY void ReSTIR::Run(
	const WaveFront::SurfaceData* const a_CurrentPixelData,
	const WaveFront::SurfaceData* const a_PreviousPixelData,
	const WaveFront::TriangleLight* a_Lights,
	const unsigned a_NumLights,
	const float3& a_CameraPosition,
	const std::uint32_t a_Seed,
	const OptixTraversableHandle a_OptixSceneHandle,
	WaveFront::AtomicBuffer<WaveFront::ShadowRayData>* a_WaveFrontShadowRayBuffer,
	const WaveFront::OptixWrapper* a_OptixSystem,
	WaveFront::MotionVectorBuffer* a_MotionVectorBuffer,
	bool a_DebugPrint
)
{
	assert(m_SwapDirtyFlag && "SwapBuffers has to be called once per frame for ReSTIR to properly work.");
	assert(a_OptixSystem && "Optix System cannot be nullptr!");

	//Index of the reservoir buffers (current and temporal).
	const auto currentIndex = m_SwapChainIndex;
	const auto temporalIndex = (currentIndex == 1 ? 0 : 1);

	//The seed will be modified over time.
	auto seed = WangHash(a_Seed);

	const unsigned numPixels = m_Settings.width * m_Settings.height;
	const uint2 dimensions = uint2{ m_Settings.width, m_Settings.height };

	//TODO: take camera position and direction into account when doing RIS.
	//TODO: Also use light area.

	//Timer for measuring performance.
	Timer timer;
	Timer totalTimer;

	/*
     * Resize buffers based on the amount of lights and update data.
     * This uploads all triangle lights. May want to move this to the wavefront pipeline class and instead take the pointer from it.
     */

	//Update the CDF for the provided light sources.
	timer.reset();
	BuildCDF(a_Lights, a_NumLights);
	if(a_DebugPrint) printf("Building CDF time required: %f millis.\n", timer.measure(TimeUnit::MILLIS));

	//Fill light bags with values from the CDF.
	{
		timer.reset();
		FillLightBags(m_Settings.numLightBags, m_Settings.numLightsPerBag, static_cast<CDF*>(m_Cdf.GetDevicePtr()), static_cast<LightBagEntry*>(m_LightBags.GetDevicePtr()), a_Lights, a_Seed);
		if (a_DebugPrint) printf("Filling light bags time required: %f millis.\n", timer.measure(TimeUnit::MILLIS));
	}

	/*
	 * Pick primary samples in parallel. Store the samples in the reservoirs.
	 */
	seed = WangHash(seed);
	timer.reset();
	PickPrimarySamples(static_cast<LightBagEntry*>(m_LightBags.GetDevicePtr()), static_cast<Reservoir*>(m_Reservoirs[currentIndex].GetDevicePtr()), m_Settings, a_CurrentPixelData, seed);
	if (a_DebugPrint) printf("Picking primary samples time required: %f millis.\n", timer.measure(TimeUnit::MILLIS));

	/*
	 * Generate shadow rays for each reservoir and resolve them.
	 * If a shadow ray is occluded, the reservoirs weight is set to 0.
	 */
	const auto numReservoirs = numPixels * m_Settings.numReservoirsPerPixel;
	timer.reset();
	const unsigned int numRaysGenerated = GenerateReSTIRShadowRays(&m_ShadowRays, static_cast<Reservoir*>(m_Reservoirs[currentIndex].GetDevicePtr()), a_CurrentPixelData, numReservoirs);
	if (a_DebugPrint) printf("ReSTIR Shadow Ray Generation time required: %f millis.\n", timer.measure(TimeUnit::MILLIS));

	//Parameters for optix launch.
	WaveFront::OptixLaunchParameters params;
	params.m_TraversableHandle = a_OptixSceneHandle;
	params.m_Reservoirs = static_cast<Reservoir*>(m_Reservoirs[currentIndex].GetDevicePtr());
	params.m_ReSTIRShadowRayBatch = m_ShadowRays.GetDevicePtr<WaveFront::AtomicBuffer<RestirShadowRay>>();
	params.m_MinMaxDistance = { 0.1f, 5000.f };	//TODO use actual numbers but idk which ones are okay ish?
	params.m_ResolutionAndDepth = make_uint3(m_Settings.width, m_Settings.height, 1);
	params.m_TraceType = WaveFront::RayType::RESTIR_RAY;

	//Tell Optix to resolve all shadow rays, which sets reservoir weight to 0 when occluded.
	if(numRaysGenerated > 0)
	{
		timer.reset();
		a_OptixSystem->TraceRays(numRaysGenerated, params);
		if (a_DebugPrint) printf("Tracing ReSTIR shadow rays time required: %f millis.\n", timer.measure(TimeUnit::MILLIS));
	}

	/*
     * Temporal sampling where reservoirs are combined with those of the previous frame.
     */
	if (m_Settings.enableTemporal)
	{
		seed = WangHash(seed);
		timer.reset();
		TemporalNeighbourSampling(
			m_Reservoirs[currentIndex].GetDevicePtr<Reservoir>(),
			m_Reservoirs[temporalIndex].GetDevicePtr<Reservoir>(),
			a_CurrentPixelData,
			a_PreviousPixelData,
			seed,
			dimensions,
			a_MotionVectorBuffer
		);
		if (a_DebugPrint) printf("Temporal sampling time required: %f millis.\n", timer.measure(TimeUnit::MILLIS));
	}

	/*
	 * Spatial sampling where neighbouring reservoirs are combined.
	 */
	if(m_Settings.enableSpatial)
	{
		seed = WangHash(seed);
		timer.reset();
		SpatialNeighbourSampling(
			static_cast<Reservoir*>(m_Reservoirs[currentIndex].GetDevicePtr()),
			static_cast<Reservoir*>(m_Reservoirs[2].GetDevicePtr()), 
			a_CurrentPixelData,
			seed,
			dimensions
		);
		if (a_DebugPrint) printf("Spatial sampling time required: %f millis.\n", timer.measure(TimeUnit::MILLIS));
	}

	timer.reset();
	GenerateWaveFrontShadowRays(
		static_cast<Reservoir*>(m_Reservoirs[currentIndex].GetDevicePtr()),
		a_CurrentPixelData,
		a_WaveFrontShadowRayBuffer,
		numPixels
	);
	if (a_DebugPrint) printf("Generating wavefront shadow rays time required: %f millis.\n", timer.measure(TimeUnit::MILLIS));

	//Ensure CUDA is done executing now.
	cudaDeviceSynchronize();

	printf("Total ReSTIR runtime: %f millis.\n", totalTimer.measure(TimeUnit::MILLIS));


	//Ensure that swap buffers is called.
	m_SwapDirtyFlag = false;
}

void ReSTIR::BuildCDF(const WaveFront::TriangleLight* a_Lights, const unsigned a_NumLights)
{
	/*
     * Resize buffers based on the amount of lights and update data.
     * This uploads all triangle lights. May want to move this to the wavefront pipeline class and instead take the pointer from it.
     */
     //CDF
	{
		const auto cdfNeededSize = sizeof(CDF) + (a_NumLights * sizeof(float));
		//Allocate enough memory for the CDF struct and the fixed sum entries.
		if (m_Cdf.GetSize() < cdfNeededSize)
		{
			m_Cdf.Resize(cdfNeededSize);
		}

		//Insert the light data in the CDF.
		FillCDF(static_cast<CDF*>(m_Cdf.GetDevicePtr()), a_Lights, a_NumLights);
	}
}

void ReSTIR::SwapBuffers()
{
	m_SwapDirtyFlag = true;
	++m_SwapChainIndex;

	//Jacco had this really cool bitflag operation for this but I'll just do it like this for now because I can't find it.
	if(m_SwapChainIndex >= 2)
	{
		m_SwapChainIndex = 0;
	}
}

CDF* ReSTIR::GetCdfGpuPointer() const
{
	return static_cast<CDF*>(m_Cdf.GetDevicePtr());
}

size_t ReSTIR::GetExpectedGpuRamUsage(const ReSTIRSettings& a_Settings, size_t a_NumLights) const
{
	const size_t reservoirSize = static_cast<size_t>(a_Settings.width) * static_cast<size_t>(a_Settings.height) * a_Settings.numReservoirsPerPixel * sizeof(Reservoir) * 3;
	const size_t cdfSize = a_NumLights * sizeof(float);
	const size_t lightBagSize = sizeof(LightBagEntry) * a_Settings.numLightsPerBag * a_Settings.numLightBags;
	const size_t shadowRaySize = sizeof(WaveFront::AtomicBuffer<RestirShadowRay>) + (static_cast<size_t>(a_Settings.width) * static_cast<size_t>(a_Settings.height) * a_Settings.numReservoirsPerPixel * sizeof(RestirShadowRay));

	return reservoirSize + cdfSize + lightBagSize + shadowRaySize;
}

size_t ReSTIR::GetAllocatedGpuMemory() const
{
	return
		m_Reservoirs[0].GetSize()
		+ m_Reservoirs[1].GetSize()
		+ m_Reservoirs[2].GetSize()
		+ m_Cdf.GetSize()
		+ m_ShadowRays.GetSize()
		+ m_LightBags.GetSize()
    ;
}
#endif
