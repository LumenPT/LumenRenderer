#pragma once

#include <nlohmann/json.hpp>
#include <filesystem>

class AppConfiguration
{

public:

    AppConfiguration();
    ~AppConfiguration() = default;

    
    bool Load(const std::filesystem::path& a_ConfigFile, bool a_CreateIfNotExists = false, bool a_CreateIfNotComplete = false, bool a_Overwrite = false);

    static const AppConfiguration& GetInstanceConst();
    static AppConfiguration& GetInstance();

    static bool CreateDefault(const std::filesystem::path& a_ConfigFile, bool a_Overwrite = false);

    const std::filesystem::path& GetDirectoryAssets() const;
    const std::filesystem::path& GetDirectoryShaders() const;
    const std::filesystem::path& GetDirectoryModels() const;

    const std::filesystem::path& GetFileShaderSolids() const;
    const std::filesystem::path& GetFileShaderVolumetrics() const;

    const std::filesystem::path& GetDefaultModel() const;
    const bool HasDefaultModel() const;

private:

    static bool VerifyComplete(nlohmann::json a_ConfigFile);



    std::filesystem::path m_AssetsDirectory;
    std::filesystem::path m_ShaderDirectory;
    std::filesystem::path m_ModelDirectory;

    std::filesystem::path m_ShadersSolids;
    std::filesystem::path m_ShadersVolumetrics;
    std::filesystem::path m_DefaultModel;

    bool m_Loaded;

    static AppConfiguration* s_Instance;

};