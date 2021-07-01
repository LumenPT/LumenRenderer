#include "CudaUtilities.h"
#include "Timer.h"
#ifdef WAVEFRONT
#include "ReSTIR.h"

#include "Lumen/Renderer/LumenRenderer.h"

#include <cassert>
#include <cuda_runtime.h>

#include "../CUDAKernels/RandomUtilities.cuh"
#include "../Shaders/CppCommon/WaveFrontDataStructs/AtomicBuffer.h"
#include "OptixWrapper.h"

void ReSTIR::Initialize(const ReSTIRSettings& a_Settings)
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
		//Multiply by two because it will contain primary sample and neighbour sample rays.
		const unsigned size = static_cast<unsigned>(m_Settings.width) * static_cast<unsigned>(m_Settings.height) * ReSTIRSettings::numReservoirsPerPixel * 2u;
		WaveFront::CreateAtomicBuffer<RestirShadowRay>(&m_ShadowRays, size);
	}

	//Reservoirs
	{
		//Reserve enough memory for both the front and back buffer to contain all reservoirs.
		//The 4 represents one buffer for primary samples, temporal samples, and two for neighbours.
		const size_t numReservoirs = static_cast<size_t>(m_Settings.width) * static_cast<size_t>(m_Settings.height) * m_Settings.numReservoirsPerPixel * 4;  
		const size_t size = numReservoirs * sizeof(Reservoir);
		m_Reservoirs.Resize(size);
		ResetReservoirs(numReservoirs, static_cast<Reservoir*>(m_Reservoirs.GetDevicePtr()));
		CHECKLASTCUDAERROR;
		
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

void ReSTIR::Run(
	const WaveFront::SurfaceData* const a_CurrentPixelData,
	const WaveFront::SurfaceData* const a_PreviousPixelData,
	const cudaSurfaceObject_t a_MotionVectorBuffer,
	const WaveFront::OptixWrapper* const a_OptixWrapper,
	const OptixTraversableHandle a_OptixSceneHandle,
	const std::uint32_t a_Seed,
	const MemoryBuffer* const a_Lights,
	std::array<cudaSurfaceObject_t, static_cast<unsigned>(WaveFront::LightChannel::NUM_CHANNELS)> a_OutputBuffer,
    FrameStats& a_FrameStats, bool a_DebugPrint
)
{
	//TODO:
	/*
	 * - Combine neighbour samples, but not with current one. Use buffer 2 and 3.
	 * - Generate shadow rays for neighbour reservoirs and then shade.
	 * - Generate shadow rays for primary reservoirs and then shade.
	 * - Shade temporal reservoirs always.
	 *
	 * - Now combine current, temporal and neighbour reservoirs.
	 */
	
	assert(m_SwapDirtyFlag && "SwapBuffers has to be called once per frame for ReSTIR to properly work.");
	assert(a_OptixWrapper && "Optix System cannot be nullptr!");

	//The stride in the reservoir buffer between the different buffers in terms of elements.
	const size_t bufferStride = static_cast<size_t>(m_Settings.width) * static_cast<size_t>(m_Settings.height) * m_Settings.numReservoirsPerPixel;

	//The offsets into the buffer for each reservoir set.
	Reservoir* reservoirPointers[4]
	{
		&m_Reservoirs.GetDevicePtr<Reservoir>()[bufferStride * 0],
		&m_Reservoirs.GetDevicePtr<Reservoir>()[bufferStride * 1],
		&m_Reservoirs.GetDevicePtr<Reservoir>()[bufferStride * 2],
		&m_Reservoirs.GetDevicePtr<Reservoir>()[bufferStride * 3],
	};

	//Index of the reservoir buffers (current and temporal).
	const auto currentIndex = m_SwapChainIndex;
	const auto temporalIndex = (currentIndex == 1 ? 0 : 1);

	//The seed will be modified over time.
	auto seed = WangHash(a_Seed);

	const unsigned numPixels = m_Settings.width * m_Settings.height;
	const uint2 dimensions = uint2{ m_Settings.width, m_Settings.height };

	//TODO: take camera position and direction into account when doing RIS.

	//Timer for measuring performance.
	Timer timer;
	Timer totalTimer;

	/*
     * Resize buffers based on the amount of lights and update data.
     * This uploads all triangle lights. May want to move this to the wavefront pipeline class and instead take the pointer from it.
     */

	//Update the CDF for the provided light sources.
	timer.reset();
	BuildCDF(a_Lights);
	if (a_DebugPrint)
	{
		auto size = WaveFront::GetAtomicCounter<WaveFront::TriangleLight>(a_Lights);
		printf("Building CDF time required: %f millis.\nNumLights: %u\n", timer.measure(TimeUnit::MILLIS), size);
		CHECKLASTCUDAERROR;
	}
	//Fill light bags with values from the CDF.
	{
		timer.reset();
		FillLightBags(
			m_Settings.numLightBags, 
			m_Settings.numLightsPerBag, 
			static_cast<CDF*>(m_Cdf.GetDevicePtr()), 
			static_cast<LightBagEntry*>(m_LightBags.GetDevicePtr()), 
			a_Lights->GetDevicePtr<WaveFront::AtomicBuffer<WaveFront::TriangleLight>>(), 
			a_Seed);
		if (a_DebugPrint) printf("Filling light bags time required: %f millis.\n", timer.measure(TimeUnit::MILLIS));
		CHECKLASTCUDAERROR;
	}

	/*
	 * Pick primary samples in parallel. Store the samples in the reservoirs.
	 * This also resolves directly hitting lights with the camera.
	 */
	seed = WangHash(seed);
	timer.reset();
	PickPrimarySamples(static_cast<LightBagEntry*>(m_LightBags.GetDevicePtr()), reservoirPointers[currentIndex], m_Settings, a_CurrentPixelData, seed);
	CHECKLASTCUDAERROR;
    if (a_DebugPrint) printf("Picking primary samples time required: %f millis.\n", timer.measure(TimeUnit::MILLIS));

	/*
	 * Do a visibility check to eliminate bad primary samples.
	 * Then shade all the surviving samples.
	 */
	timer.reset();
	VisibilityCheck(&m_ShadowRays, reservoirPointers[currentIndex], a_CurrentPixelData, a_OptixWrapper, numPixels, a_OptixSceneHandle);
	Shade(reservoirPointers[currentIndex], dimensions.x, dimensions.y, a_OutputBuffer.at(static_cast<unsigned>(WaveFront::LightChannel::DIRECT)));
	if (a_DebugPrint) printf("Primary visibility check and shading time required: %f millis.\n", timer.measure(TimeUnit::MILLIS));
	CHECKLASTCUDAERROR;
	
	/*
     * Temporal sampling where reservoirs are combined with those of the previous frame.
     * Temporal samples are always used to shade. This happens internally with no visibility check.
     */
	if (m_Settings.enableTemporal)
	{
		seed = WangHash(seed);
		timer.reset();
		TemporalNeighbourSampling(
			reservoirPointers[currentIndex],
			reservoirPointers[temporalIndex],
			a_CurrentPixelData,
			a_PreviousPixelData,
			seed,
			dimensions,
			a_MotionVectorBuffer,
			a_OutputBuffer.at(static_cast<unsigned>(WaveFront::LightChannel::DIRECT))
		);
		if (a_DebugPrint) printf("Temporal sampling time required: %f millis.\n", timer.measure(TimeUnit::MILLIS));
		CHECKLASTCUDAERROR;
	}

	/*
	 * Spatial sampling where neighbouring reservoirs are combined.
	 * Spatial neighbours are then visibility checked and shaded.
	 */
	if(m_Settings.enableSpatial)
	{
		seed = WangHash(seed);
		timer.reset();
		Reservoir* neighbourBuffer = SpatialNeighbourSampling(
			reservoirPointers[currentIndex],
			reservoirPointers[2],
			reservoirPointers[3],
			a_CurrentPixelData,
			seed,
			dimensions
		);
		if (a_DebugPrint) printf("Spatial sampling time required: %f millis.\n", timer.measure(TimeUnit::MILLIS));
		CHECKLASTCUDAERROR;

		/*
		 * Do the second visibility check for spatial samples (greatly reduces shadow bleed).
		 */
		timer.reset();
		VisibilityCheck(&m_ShadowRays, reservoirPointers[currentIndex], a_CurrentPixelData, a_OptixWrapper, numPixels, a_OptixSceneHandle);
		Shade(reservoirPointers[currentIndex], dimensions.x, dimensions.y, a_OutputBuffer.at(static_cast<unsigned>(WaveFront::LightChannel::DIRECT)));
		if (a_DebugPrint) printf("Spatial visibility check and shading time required: %f millis.\n", timer.measure(TimeUnit::MILLIS));
		CHECKLASTCUDAERROR;
		
		/*
		 * Finally, combine the neighbour reservoirs with the current reservoirs to be used in the next frame.
		 */
		timer.reset();
		CombineReservoirBuffers(reservoirPointers[currentIndex], neighbourBuffer, a_CurrentPixelData, numPixels * ReSTIRSettings::numReservoirsPerPixel, WangHash(seed));
		if (a_DebugPrint) printf("Spatial reservoir combination time required: %f millis.\n", timer.measure(TimeUnit::MILLIS));
	}

	//Ensure CUDA is done executing now.
	cudaDeviceSynchronize();

	printf("Total ReSTIR runtime: %f millis.\n", totalTimer.measure(TimeUnit::MILLIS));

	a_FrameStats.m_Times["ReSTIR"] = totalTimer.measure(TimeUnit::MICROS);

	//Ensure that swap buffers is called.
	m_SwapDirtyFlag = false;
}

void ReSTIR::BuildCDF(const MemoryBuffer* a_Lights)
{
	/*
     * Resize buffers based on the amount of lights and update data.
     * This uploads all triangle lights. May want to move this to the wavefront pipeline class and instead take the pointer from it.
     */
     //CDF
	{
		const auto numLights = WaveFront::GetAtomicCounter<WaveFront::TriangleLight>(a_Lights);
		const auto cdfNeededSize = sizeof(CDF) + (numLights * sizeof(float));
		//Allocate enough memory for the CDF struct and the fixed sum entries.
		if (m_Cdf.GetSize() < cdfNeededSize)
		{
			m_Cdf.Resize(cdfNeededSize);

			//The CDF tree requires a base 2 size. First power of 2 bigger than or equal to the number of lights.
			const unsigned power = static_cast<unsigned>(std::ceilf(std::log2f(static_cast<float>(numLights)))) + 1u;	//Take the lowest base of 2, and add 1 for the first copy.
			m_CdfTree.Resize(static_cast<size_t>(std::pow(2u, power)) * sizeof(float));
		}

		//Insert the light data in the CDF.
		FillCDF(
			static_cast<CDF*>(m_Cdf.GetDevicePtr()),
			m_CdfTree.GetDevicePtr<float>(),
			a_Lights->GetDevicePtr<WaveFront::AtomicBuffer<WaveFront::TriangleLight>>(), 
			numLights);
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
	const size_t reservoirSize = static_cast<size_t>(a_Settings.width) * static_cast<size_t>(a_Settings.height) * a_Settings.numReservoirsPerPixel * sizeof(Reservoir) * 4;
	const size_t cdfSize = a_NumLights * sizeof(float);
	const size_t lightBagSize = sizeof(LightBagEntry) * a_Settings.numLightsPerBag * a_Settings.numLightBags;
	const size_t shadowRaySize = sizeof(WaveFront::AtomicBuffer<RestirShadowRay>) + (static_cast<size_t>(a_Settings.width) * static_cast<size_t>(a_Settings.height) * a_Settings.numReservoirsPerPixel * sizeof(RestirShadowRay));

	return reservoirSize + cdfSize + lightBagSize + shadowRaySize;
}

void ReSTIR::VisibilityCheck(MemoryBuffer* a_ShadowRayAtomicBuffer, Reservoir* a_Reservoirs,
	const WaveFront::SurfaceData* a_SurfaceData, const WaveFront::OptixWrapper* a_OptixWrapper, unsigned a_NumPixels, OptixTraversableHandle a_OptixSceneHandle)
{
	/*
	 * Generate shadow rays for each reservoir and resolve them.
	 * If a shadow ray is occluded, the reservoirs weight is set to 0.
	 */
	const auto numReservoirs = a_NumPixels * ReSTIRSettings::numReservoirsPerPixel;
	const unsigned int numRaysGenerated = GenerateReSTIRShadowRays(a_ShadowRayAtomicBuffer, a_Reservoirs, a_SurfaceData, numReservoirs);
	CHECKLASTCUDAERROR;

	//Tell Optix to resolve all shadow rays, which sets reservoir weight to 0 when occluded.
	if (numRaysGenerated > 0)
	{
		//Parameters for optix launch.
		WaveFront::OptixLaunchParameters params{};
		params.m_TraversableHandle = a_OptixSceneHandle;
		params.m_Reservoirs = a_Reservoirs;
		params.m_ReSTIRShadowRayBatch = a_ShadowRayAtomicBuffer->GetDevicePtr<WaveFront::AtomicBuffer<RestirShadowRay>>();
		params.m_MinMaxDistance = { 0.1f, 10000.f };	//TODO use actual numbers but idk which ones are okay ish?
		params.m_ResolutionAndDepth = make_uint3(m_Settings.width, m_Settings.height, 1);
		params.m_TraceType = WaveFront::RayType::RESTIR_RAY;
		
		a_OptixWrapper->TraceRays(numRaysGenerated, params);
		CHECKLASTCUDAERROR;
	}
}

size_t ReSTIR::GetAllocatedGpuMemory() const
{
	return
		m_Reservoirs.GetSize()
		+ m_Cdf.GetSize()
		+ m_ShadowRays.GetSize()
		+ m_LightBags.GetSize()
    ;
}
#endif
