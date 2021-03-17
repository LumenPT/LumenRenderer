#pragma once

#ifdef WAVEFRONT
#include "WaveFrontRenderer.h"
#else
#include "OptiXRenderer.h"
#endif

//This file is to include the right renderer header file when using things like PTServiceLocator.
//LumenPT.h can not be used because of the g_OptixFunctionTable that gets redefined if you include it multiple times.