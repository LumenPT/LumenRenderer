#pragma once

#ifndef LUMENPT_H
#define LUMENPT_H

#include "Optix/optix_function_table_definition.h"

#ifdef WAVEFRONT

#include "Framework/WaveFrontRenderer.h"

using LumenPT = WaveFront::WaveFrontRenderer;

#else

#include "Framework/OptiXRenderer.h"

using LumenPT = OptiXRenderer;

#endif

#endif