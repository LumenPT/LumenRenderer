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

    // Constructs the scene out of the provided scene data. This invalidates a_SceneData for future use.
    PTScene(LumenRenderer::SceneData& a_SceneData, PTServiceLocator& a_ServiceLocator);
    ~PTScene(){};

    // Adds a new static geometry mesh to the scene
    Lumen::MeshInstance* AddMesh() override;
    // Adds a new volumete instance to the scene
    Lumen::VolumeInstance* AddVolume() override;

    // Clears all instances from the scene
    void Clear() override;

    // Sets up the scene for an update next time it is requested
    void MarkSceneForUpdate();

    // Returns the Optix acceleration structure handle for the entire scene
    OptixTraversableHandle GetSceneAccelerationStructure();

    // Updates the instance acceleration structure of the scene
    void UpdateSceneAccelerationStructure();


private:
    // Reference to the path tracer service locator. Necessary when updating the scene acceleration structure
    PTServiceLocator& m_Services;

    // Handle for the acceleration structure of the scene
    std::unique_ptr<class AccelerationStructure> m_SceneAccelerationStructure;
    // Flag to track if the acceleration structure needs
    // to be rebuilt because a new instance was added, or an existing instance was transformed
    bool m_AccelerationStructureDirty;
};
