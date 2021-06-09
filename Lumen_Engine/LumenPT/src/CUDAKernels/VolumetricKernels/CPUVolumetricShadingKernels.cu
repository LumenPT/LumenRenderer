#include "CPUVolumetricShadingKernels.cuh"
#include "GPUVolumetricShadingKernels.cuh"

CPU_ONLY void ExtractVolumetricData(
    unsigned a_NumIntersections, 
    const WaveFront::AtomicBuffer<WaveFront::VolumetricIntersectionData>* a_IntersectionData, 
    const WaveFront::AtomicBuffer<WaveFront::IntersectionRayData>* a_Rays, 
    WaveFront::VolumetricData* a_Output,
    uint2 a_Resolution,
    SceneDataTableAccessor* a_SceneDataTable)
{

    const int blockSize = 256;
    const int numBlocks = (a_NumIntersections + blockSize - 1) / blockSize;

    ExtractVolumetricDataGpu <<<numBlocks, blockSize >> > (a_NumIntersections, a_Rays, a_IntersectionData, a_Output, a_Resolution, a_SceneDataTable);

}
