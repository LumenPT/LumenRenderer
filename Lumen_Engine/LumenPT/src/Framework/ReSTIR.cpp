#include "ReSTIR.h"

#include <cassert>
#include <cuda_runtime.h>

//TODO include this but it breaks because of redefinitions.
//#include "../Shaders/CppCommon/WaveFrontDataStructs.h"


CPU_ONLY void ReSTIR::Initialize(const ReSTIRSettings& a_Settings)
{
	m_Settings = a_Settings;

	//Ensure correct configuration.
	assert(m_Settings.width != 0 && m_Settings.height != 0 && "ReSTIR requires screen dimensions to be non-zero positive values.");
	assert(m_Settings.numLightsPerBag > 0 && "Num lights per bag needs to be at least 1.");
	assert(m_Settings.numPrimarySamples > 0 && "Num primary samples needs to be at least 1.");
	assert(m_Settings.numReservoirsPerPixel > 0 && "Num reservoirs per pixel needs to be at least 1.");
	assert(m_Settings.numSpatialIterations > 0 && "The amount of spatial iterations needs to be at least 1.");
	assert(m_Settings.numSpatialSamples > 0 && "The amount of spatial samples needs to be at least 1.");
	assert(m_Settings.pixelGridSize > 0 && "The pixel grid size needs to be at least 1 by 1 pixels.");
	assert(m_Settings.spatialSampleRadius > 0 && "The spatial sample radius needs to be at least 1 pixel.");

	//Initialize the buffers required.

	//Shadow rays.
	{
		//At most one shadow ray per reservoir. Always resize even when big enough already, because this is initialization so it should not happen often.
		const size_t shadowRaySize = 1;//sizeof(WaveFront::ShadowRayData);//TODO enable
		const size_t size = static_cast<size_t>(m_Settings.width) * static_cast<size_t>(m_Settings.height) * m_Settings.numReservoirsPerPixel * shadowRaySize;
		m_ShadowRays.Resize(size);

		//TODO ensure that these rays are initialized every time for every frame so that invalid rays are not traced.
		//TODO the shadow rays should contain the reservoir index (x y and z).
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

	//Wait for CUDA to finish executing.
	cudaDeviceSynchronize();
}

CPU_ONLY void ReSTIR::Run(const WaveFront::IntersectionBuffer* const a_CurrentIntersections,
    const WaveFront::IntersectionBuffer* const a_PreviousIntersections, const std::vector<TriangleLight>& a_Lights)
{
	assert(m_SwapDirtyFlag && "SwapBuffers has to be called once per frame for ReSTIR to properly work.");

	/*
	 * Resize buffers based on the amount of lights and update data.
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
	{
	    //Light bag buffer containing indices of lights. One bag per grid cell.
		const auto numGridCellsX = std::ceilf(m_Settings.width / static_cast<float>(m_Settings.pixelGridSize));
		const auto numGridCellsY = ceilf(m_Settings.height / static_cast<float>(m_Settings.pixelGridSize));
		const size_t numLightBags = static_cast<size_t>(numGridCellsX) * static_cast<size_t>(numGridCellsY);
            
		const size_t size = sizeof(int) * m_Settings.numLightsPerBag * numLightBags;
		if (m_LightBags.GetSize() < size)
		{
			m_LightBags.Resize(size);
		}
	}
	//TODO CDF

	//TODO run algorithm.


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