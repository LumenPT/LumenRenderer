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
		WaveFront::SurfaceData* a_CurrentPixelData,
		WaveFront::SurfaceData* a_PreviousPixelData,
		const std::vector<WaveFront::TriangleLight>& a_Lights,
	    const float3 a_CameraPosition,
		const std::uint32_t a_Seed,
		const OptixTraversableHandle a_OptixSceneHandle,
		WaveFront::AtomicBuffer<WaveFront::ShadowRayData>* a_WaveFrontShadowRayBuffer,
        WaveFront::OptixWrapper* a_OptixSystem
	);

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
	MemoryBuffer m_Atomics;		//Buffer to store atomic counters in.

	//Memory buffers that need to be temporally available.
	MemoryBuffer m_Reservoirs[3];	//Reservoir buffers per frame. 0, 1 = swap chain of reservoir buffers. 2 = spatial swap buffer.
};