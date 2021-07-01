#include "AppConfiguration.h"
#include "AppConfigKeys.h"
#include "AppConfigDefaults.h"

#include <fstream>
#include <iostream>

using Json = nlohmann::json;

AppConfiguration::AppConfiguration()
    :
m_AssetsDirectory(),
m_ShadersSolids(),
m_ShadersVolumetrics(),
m_Loaded(false)
{}



bool AppConfiguration::Load(const std::filesystem::path& a_ConfigFile, bool a_CreateIfNotExists, bool a_CreateIfNotComplete, bool a_Overwrite)
{


    if (m_Loaded && !a_Overwrite)
    {
        return false;
    }

    if (!std::filesystem::exists(a_ConfigFile))
    {
        if(!a_CreateIfNotExists)
        {
            printf("Error: No config file exists with path: %s !", a_ConfigFile.string().c_str());
            return false;
        }
        else
        {
            //Create a new default config at the given filepath, should not try to overwrite as it should not exist.
            if(!CreateDefault(a_ConfigFile, false))
            {
                printf("Error: Could not create config file with path %s !", a_ConfigFile.string().c_str());
                return false;
            }
        }
    }

    std::ifstream configFile(a_ConfigFile, std::ifstream::in);

    if (!configFile.is_open())
    {
        printf("Error: Could not open config file with path: %s !", a_ConfigFile.string().c_str());
        return false;
    }

    Json jsonConfig;
    configFile >> jsonConfig;

    if (!VerifyComplete(jsonConfig))
    {
        if(a_CreateIfNotComplete)
        {
            if(!CreateDefault(a_ConfigFile, true))
            {
                printf("Error: Could not create config file with path %s !", a_ConfigFile.string().c_str());
                return false;
            }
            else
            {

                configFile.open(a_ConfigFile, std::ifstream::in);

                if (!configFile.is_open())
                {
                    printf("Error: Could not open config file with path: %s !", a_ConfigFile.string().c_str());
                    return false;
                }

                configFile >> jsonConfig;

            }
        }
        else
        {
            printf("Error: Existing config does not contain all necessary keys !");
            return false;
        }
    }

    auto directories = jsonConfig.at(KEYS_Directories);
    auto files = jsonConfig.at(KEYS_Files);

    m_AssetsDirectory = directories.at(KEYS_AssetDir).get<std::string>();
    m_ShaderDirectory = directories.at(KEYS_ShaderDir).get<std::string>();
    m_ModelDirectory = directories.at(KEYS_ModelDir).get<std::string>();

    m_ShadersSolids = files.at(KEYS_ShaderSolids).get<std::string>();
    m_ShadersVolumetrics = files.at(KEYS_ShaderVolumetrics).get<std::string>();
    m_DefaultModel = files.at(KEYS_DefaultModel).get<std::string>();

    return true;

}

bool AppConfiguration::VerifyComplete(Json a_ConfigFile)
{

    bool containsAllKeys = true;

    containsAllKeys &= a_ConfigFile.contains(KEYS_Directories);
    containsAllKeys &= a_ConfigFile.contains(KEYS_Files);

    if (containsAllKeys)
    {
        const auto directories = a_ConfigFile.at(KEYS_Directories);
        const auto files = a_ConfigFile.at(KEYS_Files);

        containsAllKeys &= directories.contains(KEYS_AssetDir);
        containsAllKeys &= directories.contains(KEYS_ShaderDir);
        containsAllKeys &= directories.contains(KEYS_ModelDir);

        containsAllKeys &= files.contains(KEYS_ShaderSolids);
        containsAllKeys &= files.contains(KEYS_ShaderVolumetrics);
        containsAllKeys &= files.contains(KEYS_DefaultModel);

        return containsAllKeys;

    }
    else return containsAllKeys;

}

const std::filesystem::path& AppConfiguration::GetDirectoryAssets() const
{

    return m_AssetsDirectory;

}

const std::filesystem::path& AppConfiguration::GetDirectoryShaders() const
{

    return m_ShaderDirectory;

}

const std::filesystem::path& AppConfiguration::GetDirectoryModels() const
{

    return m_ModelDirectory;

}

const std::filesystem::path& AppConfiguration::GetFileShaderSolids() const
{

    return m_ShadersSolids;

}

const std::filesystem::path& AppConfiguration::GetFileShaderVolumetrics() const
{

    return m_ShadersVolumetrics;

}

const std::filesystem::path& AppConfiguration::GetDefaultModel() const
{

    return m_DefaultModel;

}

const bool AppConfiguration::HasDefaultModel() const
{

    return (!m_DefaultModel.empty() && 
             m_DefaultModel.has_extension() && 
            (m_DefaultModel.extension() == ".gltf" || m_DefaultModel.extension() == ".glb"));

}



AppConfiguration* AppConfiguration::s_Instance = nullptr;

const AppConfiguration& AppConfiguration::GetInstanceConst()
{

    return GetInstance();

}

AppConfiguration& AppConfiguration::GetInstance()
{

    if (!s_Instance)
    {
        s_Instance = new AppConfiguration();
    }

    return *s_Instance;

}

bool AppConfiguration::CreateDefault(const std::filesystem::path& a_ConfigFile, bool a_Overwrite)
{

    if (std::filesystem::exists(a_ConfigFile) && !a_Overwrite)
    {
        printf("Error: Could not create default file with path %s !", a_ConfigFile.string().c_str());
        return false;
    }

    std::ofstream configFile(a_ConfigFile, std::ofstream::out);

    if(!configFile.is_open())
    {
        printf("Error: Could not open config file with path %s !", a_ConfigFile.string().c_str());
        return false;
    }

    const Json jsonConfig =
    {
        { KEYS_Directories,
            {
                {KEYS_AssetDir, DEFAULTS_AssetDir},
                {KEYS_ShaderDir, DEFAULTS_ShaderDir},
                {KEYS_ModelDir, DEFAULTS_ModelDir}
            }
        },
        { KEYS_Files,
            {
                {KEYS_ShaderSolids, DEFAULTS_ShaderProgramFile_Solids},
                {KEYS_ShaderVolumetrics, DEFAULTS_ShaderProgramFile_Volumetrics},
                {KEYS_DefaultModel, DEFAULTS_DefaultModel}
            }
        }
    };

    configFile << std::setw(4) << jsonConfig << std::endl;
    configFile.close();

    return !configFile.fail();

}