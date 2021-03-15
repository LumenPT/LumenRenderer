#include "ILumenScene.h"

Lumen::MeshInstance* Lumen::ILumenScene::AddMesh()
{
    m_MeshInstances.push_back(std::make_unique<MeshInstance>());
    return m_MeshInstances.back().get();
}

Lumen::VolumeInstance* Lumen::ILumenScene::AddVolume()
{
    m_VolumeInstances.push_back(std::make_unique<VolumeInstance>());
    return m_VolumeInstances.back().get();
}

void Lumen::ILumenScene::Clear()
{
    m_VolumeInstances.clear();
    m_MeshInstances.clear();
}
