#pragma once
#include "ModelLoading/MeshInstance.h"

#include "Optix/optix_types.h"

class PTScene;
class PTServiceLocator;

class PTMeshInstance : public Lumen::MeshInstance
{
public:
    PTMeshInstance(PTServiceLocator& a_ServiceLocator);
    PTMeshInstance(const Lumen::MeshInstance& a_Instance, PTServiceLocator& a_ServiceLocator);

    void DependencyCallback();
    // Function called under the hood to establish a connection between the instance and the scene
    // Necessary for the acceleration structures to be rebuilt when the transform changes
    void SetSceneRef(PTScene* a_SceneRef);

    void SetMesh(std::shared_ptr<Lumen::ILumenMesh> a_Mesh) override;

    PTScene* m_SceneRef;
    PTServiceLocator& m_Services;
};
