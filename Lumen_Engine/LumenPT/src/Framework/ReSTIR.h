#pragma once
#include <cinttypes>


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
	/*
	 * Initialize the ReSTIR required buffers for the given screen dimensions.
	 */
	void Initialize(const ReSTIRSettings& a_Settings);

	/*
	 * Run ReSTIR.
	 */
	void Run();


private:
	ReSTIRSettings m_Settings;
};