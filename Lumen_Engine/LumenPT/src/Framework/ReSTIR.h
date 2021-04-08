#pragma once
#include <cinttypes>

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
	CPU_ONLY void Initialize(const ReSTIRSettings& a_Settings);

	/*
	 * Run ReSTIR.
	 */
	CPU_ONLY void Run(
		const WaveFront::SurfaceData * const a_CurrentPixelData,
		const WaveFront::SurfaceData * const a_PreviousPixelData,
		const WaveFront::TriangleLight* a_Lights,
		const unsigned a_NumLights,
	    const float3& a_CameraPosition,
		const std::uint32_t a_Seed,
		const OptixTraversableHandle a_OptixSceneHandle,
		WaveFront::AtomicBuffer<WaveFront::ShadowRayData>* a_WaveFrontShadowRayBuffer,
        const WaveFront::OptixWrapper* a_OptixSystem,
		WaveFront::MotionVectorBuffer* a_MotionVectorBuffer,
		bool a_DebugPrint = false
	);

	/*
	 * Update the CDF for the given light sources.
	 */
	CPU_ONLY void BuildCDF(const WaveFront::TriangleLight* a_Lights, const unsigned a_NumLights);

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
	 * Get the size in bytes of all allocated GPU memory owned by ReSTIR.
	 */
	size_t GetAllocatedGpuMemory() const;


private:
	ReSTIRSettings m_Settings;
	int m_SwapChainIndex;
	bool m_SwapDirtyFlag;	//Dirty flag to assert if someone forgets to swap the buffers.

	//Memory buffers only used in the current frame.
	MemoryBuffer m_Cdf;			//The CDF which is the size of a CDF entry times the amount of lights.
	MemoryBuffer m_LightBags;	//All light bags as a single array. Size of num light bags * size of light bag * light index or something.
	MemoryBuffer m_ShadowRays;	//Buffer for each shadow ray in a frame. Size of screen dimensions * ray size.

	//Memory buffers that need to be temporally available.
	MemoryBuffer m_Reservoirs[3];	//Reservoir buffers per frame. 0, 1 = swap chain of reservoir buffers. 2 = spatial swap buffer.
};