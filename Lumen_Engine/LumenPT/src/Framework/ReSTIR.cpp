#include "ReSTIR.h"

#include <cassert>
#include <cuda_runtime.h>


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

		//TODO ensure that these rays are initialized every time for every frame so that invalid rays are not traced.
		//TODO the shadow rays should contain the reservoir index (x y and z).
	}

	//Pixel data caching
	{
		const size_t numPixels = m_Settings.width * m_Settings.height;
		const size_t size = numPixels * sizeof(PixelData);
		m_Pixels[0].Resize(size);
		m_Pixels[1].Resize(size);
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
	const WaveFront::IntersectionData* const a_CurrentIntersections,
	const WaveFront::RayData* const a_RayBuffer,
	const std::vector<TriangleLight>& a_Lights,
	const float3 a_CameraPosition
)
{
	assert(m_SwapDirtyFlag && "SwapBuffers has to be called once per frame for ReSTIR to properly work.");

	//Index of the reservoir buffers (current and temporal).
	auto currentIndex = m_SwapChainIndex;
	auto temporalIndex = currentIndex == 1 ? 0 : 1;

	/*
	 * Resize buffers based on the amount of lights and update data.
	 * This uploads all triangle lights. May want to move this to the wavefront pipeline class and instead take the pointer from it.
	 */
	{
		//Light buffer
		const size_t size = sizeof(TriangleLight) * a_Lights.size();
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
		FillCDF(static_cast<CDF*>(m_Cdf.GetDevicePtr()), static_cast<TriangleLight*>(m_Lights.GetDevicePtr()), m_Lights.GetSize());
	}
	//Fill light bags with values from the CDF.
	{
		FillLightBags(m_Settings.numLightBags, static_cast<CDF*>(m_Cdf.GetDevicePtr()), static_cast<LightBagEntry*>(m_LightBags.GetDevicePtr()), static_cast<TriangleLight*>(m_Lights.GetDevicePtr()));
	}

	//Pointers to the pixel data buffers.
	PixelData* currentPixelData = m_Pixels[currentIndex].GetDevicePtr<PixelData>();
	PixelData* temporalPixelData = m_Pixels[temporalIndex].GetDevicePtr<PixelData>();

	/*
	 * Pick primary samples in parallel. Store the samples in the reservoirs.
	 */
	PickPrimarySamples(a_RayBuffer, a_CurrentIntersections, static_cast<LightBagEntry*>(m_LightBags.GetDevicePtr()), static_cast<Reservoir*>(m_Reservoirs->GetDevicePtr()), m_Settings, currentPixelData);

	/*
	 * Generate shadow rays for each reservoir and resolve them.
	 * If a shadow ray is occluded, the reservoirs weight is set to 0.
	 */
	VisibilityPass(&m_Atomics, static_cast<Reservoir*>(m_Reservoirs[m_SwapChainIndex].GetDevicePtr()), m_ShadowRays.GetDevicePtr<RestirShadowRay>(), currentPixelData);

	/*
     * Temporal sampling where reservoirs are combined with those of the previous frame.
     */
	if (m_Settings.enableTemporal)
	{
		TemporalNeighbourSampling(
			m_Reservoirs[currentIndex].GetDevicePtr<Reservoir>(),
			m_Reservoirs[temporalIndex].GetDevicePtr<Reservoir>(),
			currentPixelData,
			temporalPixelData
		);
	}

	/*
	 * Spatial sampling where neighbouring reservoirs are combined.
	 */
	if(m_Settings.enableSpatial)
	{
		SpatialNeighbourSampling(
			static_cast<Reservoir*>(m_Reservoirs[currentIndex].GetDevicePtr()),
			static_cast<Reservoir*>(m_Reservoirs[2].GetDevicePtr()), 
			currentPixelData
		);
	}

	//TODO: Generate regular WaveFront shadow rays and put them in the buffer in WaveFront.
	//TODO: This will require an atomic + the buffer to be passed.
	//TODO: The contribution of the shadow rays will be scaled with the reservoir weight here.

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
