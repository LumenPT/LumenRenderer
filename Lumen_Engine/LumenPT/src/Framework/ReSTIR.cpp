#ifdef WAVEFRONT
#include "ReSTIR.h"

#include <cassert>
#include <cuda_runtime.h>

#include "../CUDAKernels/RandomUtilities.cuh"
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
		const size_t shadowRaySize = sizeof(RestirShadowRay);
		const size_t size = static_cast<size_t>(m_Settings.width) * static_cast<size_t>(m_Settings.height) * m_Settings.numReservoirsPerPixel * shadowRaySize;
		m_ShadowRays.Resize(size);
	}

	//Reservoirs
	{
		//Reserve enough memory for both the front and back buffer to contain all reservoirs.
		const size_t numReservoirs = static_cast<size_t>(m_Settings.width) * static_cast<size_t>(m_Settings.height) * m_Settings.numReservoirsPerPixel;
		const size_t size = numReservoirs * sizeof(Reservoir);
		m_Reservoirs[0].Resize(size);
		m_Reservoirs[1].Resize(size);

		//Reset both buffers.
		ResetReservoirs(numReservoirs, static_cast<Reservoir*>(m_Reservoirs[0].GetDevicePtr()));
		ResetReservoirs(numReservoirs, static_cast<Reservoir*>(m_Reservoirs[1].GetDevicePtr()));
	}

	//Light bag generation
	{
		const size_t size = sizeof(LightBagEntry) * m_Settings.numLightsPerBag * m_Settings.numLightBags;
		if (m_LightBags.GetSize() < size)
		{
			m_LightBags.Resize(size);
		}
	}

	//Atomic counter buffer
	m_Atomics.Resize(sizeof(int) * 1);

	//Wait for CUDA to finish executing.
	cudaDeviceSynchronize();
}

CPU_ONLY void ReSTIR::Run(
	WaveFront::SurfaceData* a_CurrentPixelData,
	WaveFront::SurfaceData* a_PreviousPixelData,
	const std::vector<WaveFront::TriangleLight>& a_Lights,
	const float3 a_CameraPosition,
	const std::uint32_t a_Seed,
	const OptixTraversableHandle a_OptixSceneHandle,
	WaveFront::AtomicBuffer<WaveFront::ShadowRayData>* a_WaveFrontShadowRayBuffer,
	WaveFront::OptixWrapper* a_OptixSystem
)
{
	assert(m_SwapDirtyFlag && "SwapBuffers has to be called once per frame for ReSTIR to properly work.");
	assert(a_OptixSystem && "Optix System cannot be nullptr!");

	//Index of the reservoir buffers (current and temporal).
	auto currentIndex = m_SwapChainIndex;
	auto temporalIndex = currentIndex == 1 ? 0 : 1;

	//The seed will be modified over time.
	auto seed = a_Seed;

	/*
	 * Resize buffers based on the amount of lights and update data.
	 * This uploads all triangle lights. May want to move this to the wavefront pipeline class and instead take the pointer from it.
	 */
	{
		//Light buffer
		const size_t size = sizeof(WaveFront::TriangleLight) * a_Lights.size();
		if (m_Lights.GetSize() < size)
		{
			m_Lights.Resize(size);
		}
		m_Lights.Write(&a_Lights[0], size, 0);
	}
	//CDF
	{
		//Allocate enough memory for the CDF struct and the fixed sum entries.
		m_Cdf.Resize(sizeof(CDF) + (a_Lights.size() * sizeof(float)));

		//Insert the light data in the CDF.
		FillCDF(static_cast<CDF*>(m_Cdf.GetDevicePtr()), static_cast<WaveFront::TriangleLight*>(m_Lights.GetDevicePtr()), m_Lights.GetSize());
	}
	//Fill light bags with values from the CDF.
	{
		seed = WangHash(seed);
		FillLightBags(m_Settings.numLightBags, static_cast<CDF*>(m_Cdf.GetDevicePtr()), static_cast<LightBagEntry*>(m_LightBags.GetDevicePtr()), static_cast<WaveFront::TriangleLight*>(m_Lights.GetDevicePtr()), seed);
	}

	/*
	 * Pick primary samples in parallel. Store the samples in the reservoirs.
	 */
	seed = WangHash(seed);
	PickPrimarySamples(static_cast<LightBagEntry*>(m_LightBags.GetDevicePtr()), static_cast<Reservoir*>(m_Reservoirs->GetDevicePtr()), m_Settings, a_CurrentPixelData, seed);

	/*
	 * Generate shadow rays for each reservoir and resolve them.
	 * If a shadow ray is occluded, the reservoirs weight is set to 0.
	 */
	const int numRaysGenerated = GenerateReSTIRShadowRays(&m_Atomics, static_cast<Reservoir*>(m_Reservoirs[m_SwapChainIndex].GetDevicePtr()), m_ShadowRays.GetDevicePtr<RestirShadowRay>(), a_CurrentPixelData);

	//Parameters for optix launch.
	WaveFront::OptixLaunchParameters params;
	params.m_TraversableHandle = a_OptixSceneHandle;
	params.m_Reservoirs = static_cast<Reservoir*>(m_Reservoirs[m_SwapChainIndex].GetDevicePtr());
	params.m_ReSTIRShadowRayBatch = m_ShadowRays.GetDevicePtr<WaveFront::AtomicBuffer<RestirShadowRay>>();

	//Tell Optix to resolve all shadow rays, which sets reservoir weight to 0 when occluded.
	a_OptixSystem->TraceRays(numRaysGenerated, params);

	/*
     * Temporal sampling where reservoirs are combined with those of the previous frame.
     */
	if (m_Settings.enableTemporal)
	{
		seed = WangHash(seed);
		TemporalNeighbourSampling(
			m_Reservoirs[currentIndex].GetDevicePtr<Reservoir>(),
			m_Reservoirs[temporalIndex].GetDevicePtr<Reservoir>(),
			a_CurrentPixelData,
			a_PreviousPixelData,
			seed
		);
	}

	/*
	 * Spatial sampling where neighbouring reservoirs are combined.
	 */
	if(m_Settings.enableSpatial)
	{
		seed = WangHash(seed);
		SpatialNeighbourSampling(
			static_cast<Reservoir*>(m_Reservoirs[currentIndex].GetDevicePtr()),
			static_cast<Reservoir*>(m_Reservoirs[2].GetDevicePtr()), 
			a_CurrentPixelData,
			seed
		);
	}

	GenerateWaveFrontShadowRays(
		static_cast<Reservoir*>(m_Reservoirs[currentIndex].GetDevicePtr()),
		a_CurrentPixelData,
		a_WaveFrontShadowRayBuffer
	);

	//Ensure that swap buffers is called.
	m_SwapDirtyFlag = false;
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
#endif