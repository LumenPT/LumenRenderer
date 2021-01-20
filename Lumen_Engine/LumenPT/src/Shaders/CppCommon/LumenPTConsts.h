#pragma once

#include <string>
#include "LumenPTConfig.h"

namespace LumenPTConsts
{

	const std::string gs_ConfigName = CONFIG_NAME;
	const std::string gs_ShaderPathBase = std::string(std::string(LUMENPT_BINARIES_DIRECTORY) + "\\Assets\\PrecompiledShaders\\" + gs_ConfigName + "\\");
	//const std::string gs_ShaderPathBase = std::string(std::string(LUMENPT_BINARIES_DIRECTORY) + "\\CudaPTX.dir\\Debug\\");
	const std::string gs_AssetDirectory = std::string(std::string(LUMENPT_DIRECTORY) + "\\Assets\\");

}