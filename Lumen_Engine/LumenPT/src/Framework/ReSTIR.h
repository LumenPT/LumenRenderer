#pragma once
#include <cinttypes>

#include "../CUDAKernels/ReSTIRKernels.cuh"
#include "../Shaders/CppCommon/ReSTIRData.h"
#include "MemoryBuffer.h"


namespace WaveFront {
    struct IntersectionBuffer;
}

/*
 * All configurable settings for ReSTIR.
 * The defaults are the values used in the ReSTIR paper.
 */
struct ReSTIRSettings
{
	//Screen width in pixels.
	std::uint32_t width = 0;

	//Screen height in pixels.
	std::uint32_t height = 0;

	//The amount of reservoirs used per pixel.
	std::uint32_t numReservoirsPerPixel = 5;

	//The amount of lights per light bag.
	std::uint32_t numLightsPerBag = 1000;

	//The total amount of light bags to generate.
	std::uint32_t numLightBags = 50;

	//The amount of initial samples to take each frame.
	std::uint32_t numPrimarySamples = 32;

	//The amount of spatial neighbours to consider.
	std::uint32_t numSpatialSamples = 5;

	//The maximum distance for spatial samples.
	std::uint32_t spatialSampleRadius = 30;

	//The x and y size of the pixel grid per light bag. This indirectly determines the amount of light bags.
	std::uint32_t pixelGridSize = 16;

	//The amount of spatial iterations to perform. Previous output becomes current input.
	std::uint32_t numSpatialIterations = 2;

	//Use the biased algorithm or not. When false, the unbiased algorithm is used instead.
	bool enableBiased = true;

	//Enable spatial sampling.
	bool enableSpatial = true;

	//Enable temporal sampling.
	bool enableTemporal = true;
};

/*
 * This is the main interface used to setup and run ReSTIR.
 * It owns resources allocated on the GPU to operate on.
 */
class ReSTIR
{
public:
	ReSTIR() : m_SwapChainIndex(0), m_SwapDirtyFlag(true)
    {}

	/*
	 * Initialize the ReSTIR required buffers for the given screen dimensions.
	 */
	CPU_ONLY void Initialize(const ReSTIRSettings& a_Settings);

	/*
	 * Run ReSTIR.
	 */
	CPU_ONLY void Run(const WaveFront::IntersectionBuffer* const a_CurrentIntersections, const WaveFront::IntersectionBuffer* const a_PreviousIntersections, const std::vector<TriangleLight>& a_Lights);

	/*
	 * Swap the front and back buffer. This has to be called once per frame.
	 */
	void SwapBuffers();

	/*
	 * Get a GPU pointer to the CDF.
	 */
	CDF* GetCdfGpuPointer() const;


private:
	ReSTIRSettings m_Settings;
	int m_SwapChainIndex;
	bool m_SwapDirtyFlag;	//Dirty flag to assert if someone forgets to swap the buffers.

	//Memory buffers only used in the current frame.
	MemoryBuffer m_Lights;		//All the triangle lights stored contiguously. Size of the amount of lights.
	MemoryBuffer m_Cdf;			//The CDF which is the size of a CDF entry times the amount of lights.
	MemoryBuffer m_LightBags;	//All light bags as a single array. Size of num light bags * size of light bag * light index or something.
	MemoryBuffer m_ShadowRays;	//Buffer for each shadow ray in a frame. Size of screen dimensions * ray size.

	//Memory buffers that need to be temporally available.
	MemoryBuffer m_Reservoirs[2];	//Reservoir buffers per frame.
};