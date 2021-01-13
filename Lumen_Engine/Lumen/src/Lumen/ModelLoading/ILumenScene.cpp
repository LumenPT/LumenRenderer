#include "ILumenScene.h"

Lumen::MeshInstance* Lumen::ILumenScene::AddMesh()
{
    m_MeshInstances.push_back(std::make_unique<MeshInstance>());
    return m_MeshInstances.back().get();
}
