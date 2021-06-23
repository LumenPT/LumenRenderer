#pragma once
#include "ReSTIRData.h"
#include "WaveFrontDataStructs/AtomicBuffer.h"
#include "WaveFrontDataStructs/GPUDataBuffers.h"
#include "WaveFrontDataStructs/CudaKernelParamStructs.h"
#include "WaveFrontDataStructs/WavefrontTraceMasks.h"
#include "WaveFrontDataStructs/OptixLaunchParams.h"
#include "WaveFrontDataStructs/OptixShaderStructs.h"
#include "WaveFrontDataStructs/SurfaceData.h"
#include "WaveFrontDataStructs/VolumetricData.h"
#include "WaveFrontDataStructs/LightData.h"

#define PIXEL_DATA_INDEX(PIXELX, PIXELY, WIDTH) ((PIXELY * WIDTH) + PIXELX)