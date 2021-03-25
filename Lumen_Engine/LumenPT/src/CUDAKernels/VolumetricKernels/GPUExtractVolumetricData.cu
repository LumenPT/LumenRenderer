#include "GPUVolumetricShadingKernels.cuh"
#include <device_launch_parameters.h>
#include <sutil/vec_math.h>

using namespace WaveFront;

CPU_ON_GPU void ExtractVolumetricDataGpu(
    unsigned a_NumIntersections,
    WaveFront::AtomicBuffer<WaveFront::IntersectionRayData>* a_Rays,
    WaveFront::AtomicBuffer<WaveFront::VolumetricIntersectionData>* a_IntersectionData,
    WaveFront::VolumetricData* a_OutPut,
    SceneDataTableAccessor* a_SceneDataTable)
{

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < a_NumIntersections; i += stride)
    {

        const VolumetricIntersectionData* currIntersection = a_IntersectionData->GetData(i);
        const IntersectionRayData* currRay = a_Rays->GetData(currIntersection->m_RayArrayIndex);
        unsigned int pixelIndex = currIntersection->m_PixelIndex;


        //TODO: for each intersection fill a VolumetricData struct and place in the right pixel index.
        //The struct with information will be used in the shading functions so should contain all the necessary data for this.
        //eg. incoming ray direction, entryIntersectionT, exitIntersectionT, position, etc.

        auto& output = a_OutPut[pixelIndex];
        output.m_PixelIndex = pixelIndex;
        output.m_PositionEntry = currRay->m_Origin + currRay->m_Direction * currIntersection->m_EntryT;
        output.m_PositionExit = currRay->m_Origin + currRay->m_Direction * currIntersection->m_ExitT;

    }

}