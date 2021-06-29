#pragma once
#include <string>

const std::string ConfigName = CONFIG_NAME;
const std::string DEFAULTS_AssetDir = "./Sandbox/assets/";
const std::string DEFAULTS_ShaderDir = DEFAULTS_AssetDir + "Shaders/";
const std::string DEFAULTS_ModelDir = DEFAULTS_AssetDir + "models/";

const std::string DEFAULTS_ShaderProgramFile_Solids =  ConfigName + "/WaveFrontShaders.ptx";
const std::string DEFAULTS_ShaderProgramFile_Volumetrics = ConfigName + "/volumetric_wavefront.ptx";
const std::string DEFAULTS_DefaultModel = "LowpolyRoom/scene.glb";