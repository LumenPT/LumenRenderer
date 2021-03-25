#include "GPUVolumetricShadingKernels.cuh"
#include <device_launch_parameters.h>
#include <sutil/vec_math.h>

using namespace WaveFront;

GPU_ONLY void VolumetricShadeDirect(
    const uint3 a_ResolutionAndDepth,
    const WaveFront::VolumetricData* a_VolumetricDataBuffer,
    WaveFront::AtomicBuffer<WaveFront::ShadowRayData>* const a_ShadowRays,
    const WaveFront::TriangleLight* const a_Lights,
    const unsigned int a_NumLights,
    const CDF* const a_CDF)
{

    return;

}