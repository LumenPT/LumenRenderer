#pragma once
#include "ModelLoading/ILumenScene.h"

#include "Renderer/LumenRenderer.h"

#include "Optix/optix_types.h"

#include <set>

struct PTServiceLocator;
class PTMeshInstance;

class PTScene : public Lumen::ILumenScene
{
public:

    PTScene(LumenRenderer::SceneData& a_SceneData, PTServiceLocator& a_ServiceLocator);
    ~PTScene(){};

    Lumen::MeshInstance* AddMesh() override;
    Lumen::VolumeInstance* AddVolume() override;

    void Clear() override;

    void MarkSceneForUpdate();

    OptixTraversableHandle GetSceneAccelerationStructure();

    void UpdateSceneAccelerationStructure();

    std::unique_ptr<class AccelerationStructure> m_SceneAccelerationStructure;

    bool m_AccelerationStructureDirty;

private:
    PTServiceLocator& m_Services;


};
