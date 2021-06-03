#include "lmnpch.h"
#include "LumenRenderer.h"

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

std::shared_ptr<Lumen::ILumenScene> LumenRenderer::CreateScene(SceneData a_SceneData)
{
    std::shared_ptr<Lumen::ILumenScene> scene = std::make_shared<Lumen::ILumenScene>();

    for (auto& mesh : a_SceneData.m_InstancedMeshes)
    {
        scene->m_MeshInstances.push_back(std::make_unique<Lumen::MeshInstance>(mesh));
    }

    return scene;
}
