#pragma once
#include "../../Shaders/CppCommon/CudaDefines.h"
#include "../../Shaders/CppCommon/WaveFrontDataStructs.h"

class SceneDataTableAccessor;

CPU_ONLY void ExtractVolumetricData(
    unsigned int a_NumIntersections,
    const WaveFront::AtomicBuffer<WaveFront::VolumetricIntersectionData>* a_IntersectionData,
    const WaveFront::AtomicBuffer<WaveFront::IntersectionRayData>* a_Rays,
    WaveFront::VolumetricData* a_Output,
    uint2 a_Resolution,
    SceneDataTableAccessor* a_SceneDataTable);