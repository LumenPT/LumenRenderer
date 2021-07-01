#pragma once
#include <cinttypes>
#include <cuda.h>

#include "../CUDAKernels/ReSTIRKernels.cuh"
#include "../Shaders/CppCommon/ReSTIRData.h"
#include "MemoryBuffer.h"


namespace WaveFront {
    class OptixWrapper;
    struct IntersectionBuffer;
}

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
	void Initialize(const ReSTIRSettings& a_Settings);

	/*
	 * Run ReSTIR.
	 */
	void Run(
		const WaveFront::SurfaceData* const a_CurrentPixelData,
		const WaveFront::SurfaceData* const a_PreviousPixelData,
		const cudaSurfaceObject_t a_MotionVectorBuffer,
		const WaveFront::OptixWrapper* const a_OptixWrapper,
		const OptixTraversableHandle a_OptixSceneHandle,
		const std::uint32_t a_Seed,
		const MemoryBuffer* const a_LightDataBuffer,
		std::array<cudaSurfaceObject_t, static_cast<unsigned>(WaveFront::LightChannel::NUM_CHANNELS)> a_OutputBuffer,
        struct FrameStats& a_FrameStats, bool a_DebugPrint = false
    );

	/*
	 * Update the CDF for the given light sources.
	 */
	void BuildCDF(const MemoryBuffer* a_Lights);

	/*
	 * Swap the front and back buffer. This has to be called once per frame.
	 */
	void SwapBuffers();

	/*
	 * Get a GPU pointer to the CDF.
	 */
	CDF* GetCdfGpuPointer() const;

	/*
	 * Get the expected vram usage of ReSTIR with the given settings and light count.
	 * This is allocated when ReSTIR is initialized.
	 * The Light CDF resizes based on the light count, but should have minimal impact.
	 *
	 * The returned value is in bytes.
	 */
	size_t GetExpectedGpuRamUsage(const ReSTIRSettings& a_Settings, size_t a_NumLights) const;

	/*
	 * Perform a visibility check on a set of reservoirs.
	 * This generates shadow rays which then set reservoir weights to 0 when occluded.
	 */
	void VisibilityCheck(
		MemoryBuffer* a_ShadowRayAtomicBuffer,
		Reservoir* a_Reservoirs,
		const WaveFront::SurfaceData* a_SurfaceData,
		const WaveFront::OptixWrapper* a_OptixWrapper,
		unsigned a_NumPixels,
		OptixTraversableHandle a_OptixSceneHandle);

	/*
	 * Get the size in bytes of all allocated GPU memory owned by ReSTIR.
	 */
	size_t GetAllocatedGpuMemory() const;


private:
	ReSTIRSettings m_Settings;
	int m_SwapChainIndex;
	bool m_SwapDirtyFlag;	//Dirty flag to assert if someone forgets to swap the buffers.

	//Memory buffers only used in the current frame.
	MemoryBuffer m_CdfTree;		//Buffer for tree building for the CDF. 
	MemoryBuffer m_Cdf;			//The CDF which is the size of a CDF entry times the amount of lights.
	MemoryBuffer m_LightBags;	//All light bags as a single array. Size of num light bags * size of light bag * light index or something.
	MemoryBuffer m_ShadowRays;	//Buffer for each shadow ray in a frame. Size of screen dimensions * ray size * reservoirsPerPixel.
	//MemoryBuffer m_ShadowRaysShading;	//Buffer for each shadow ray in a frame. Size of screen dimensions * ray size * reservoirsPerPixel.

	//Memory buffers that need to be temporally available.
	MemoryBuffer m_Reservoirs;	//Reservoir buffers per frame.
};