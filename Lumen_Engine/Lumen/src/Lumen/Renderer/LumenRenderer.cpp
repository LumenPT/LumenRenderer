#include "lmnpch.h"
#include "LumenRenderer.h"

std::shared_ptr<Lumen::ILumenScene> LumenRenderer::CreateScene(SceneData a_SceneData)
{
    std::shared_ptr<Lumen::ILumenScene> scene = std::make_shared<Lumen::ILumenScene>();

    for (auto& mesh : a_SceneData.m_InstancedMeshes)
    {
        scene->m_MeshInstances.push_back(std::make_unique<Lumen::MeshInstance>(mesh));
    }

    return scene;
}
