#pragma once

#ifndef LUMENPT_H
#define LUMENPT_H

#include "Optix/optix_function_table_definition.h"

#ifdef WAVEFRONT

#include "Framework/WaveFrontRenderer2WithAVengeance.h"

using LumenPT = WaveFront::WaveFrontRenderer2WithAVengeance;

#else

#include "Framework/OptiXRenderer.h"

using LumenPT = OptiXRenderer;

#endif

#endif