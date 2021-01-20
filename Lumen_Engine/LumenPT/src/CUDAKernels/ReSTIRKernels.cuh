#pragma once

#include <cuda_runtime.h>
#include "../Shaders/CppCommon/ReSTIRData.h"

__host__ void ResetReservoirs(int a_NumReservoirs, Reservoir* a_ReservoirPointer);

__global__ void ResetReservoirInternal(int a_NumReservoirs, Reservoir* a_ReservoirPointer);
