#pragma once
#include "../Cuda/device_atomic_functions.h"
#include "CudaDefines.h"
#include "ReSTIRData.h"
#include "WaveFrontDataStructs/AtomicBuffer.h"
#include "WaveFrontDataStructs/GPUDataBuffers.h"
#include "WaveFrontDataStructs/CudaKernelParamStructs.h"
#include "WaveFrontDataStructs/OptixLaunchParams.h"
#include "WaveFrontDataStructs/OptixShaderStructs.h"
#include "WaveFrontDataStructs/SurfaceDataBuffer.h"

namespace WaveFront
{
    //TODO: move this but I am lazy atm. 
    /*
     * All light channels used by WaveFront.
     */
    enum class LightChannel
    {
        DIRECT,
        INDIRECT,
        SPECULAR,
        NUM_CHANNELS
    };
}