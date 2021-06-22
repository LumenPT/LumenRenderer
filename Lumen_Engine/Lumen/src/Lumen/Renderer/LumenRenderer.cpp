#include "lmnpch.h"
#include "LumenRenderer.h"

#include <mutex>

Lumen::SceneManager::GLTFResource LumenRenderer::OpenCustomFileFormat(const std::string& a_OriginalFilePath)
{
    // By default, there is no custom file format, so we return an empty GLTFResource so that the scene manager can load the file normally.
    Lumen::SceneManager::GLTFResource res;
    res.m_Path = "";

    return res;
}

Lumen::SceneManager::GLTFResource LumenRenderer::CreateCustomFileFormat(const std::string& a_OriginalFilePath)
{
    // By default, there is no custom file format, so we return an empty GLTFResource so that the scene manager can load the file normally.
    Lumen::SceneManager::GLTFResource res;
    res.m_Path = "";

    return res;
}

std::shared_ptr<Lumen::ILumenMaterial> LumenRenderer::CreateMaterial()
{
    LumenRenderer::MaterialData matData;
    matData.m_ClearCoatRoughnessTexture = m_DefaultWhiteTexture;
    matData.m_ClearCoatTexture = m_DefaultWhiteTexture;
    matData.m_DiffuseTexture = m_DefaultDiffuseTexture;
    matData.m_EmissiveTexture = m_DefaultWhiteTexture;
    matData.m_MetallicRoughnessTexture = m_DefaultWhiteTexture;
    matData.m_NormalMap = m_DefaultNormalTexture;
    matData.m_TintTexture = m_DefaultWhiteTexture;
    matData.m_TransmissionTexture = m_DefaultWhiteTexture;
    return CreateMaterial(matData);
}

std::shared_ptr<Lumen::ILumenScene> LumenRenderer::CreateScene(SceneData a_SceneData)
{
    std::shared_ptr<Lumen::ILumenScene> scene = std::make_shared<Lumen::ILumenScene>();

    for (auto& mesh : a_SceneData.m_InstancedMeshes)
    {
        scene->m_MeshInstances.push_back(std::make_unique<Lumen::MeshInstance>(mesh));
    }

    return scene;
}

void LumenRenderer::CreateDefaultResources()
{
    uchar4 whitePixel = { 255,255,255,255 };
    uchar4 diffusePixel{ 255, 255, 255, 255 };
    uchar4 normal = { 128, 128, 255, 0 };
    m_DefaultWhiteTexture = CreateTexture(&whitePixel, 1, 1, false);
    m_DefaultDiffuseTexture = CreateTexture(&diffusePixel, 1, 1, false);
    m_DefaultNormalTexture = CreateTexture(&normal, 1, 1, false);
}

FrameStats LumenRenderer::GetLastFrameStats()
{
    std::lock_guard<std::mutex> lk(m_FrameStatsMutex);

    return m_LastFrameStats;
}
