#pragma once
#include "ModelLoading/VolumeInstance.h"

#include "Optix/optix_types.h"

class PTScene;
struct PTServiceLocator;

class PTVolumeInstance : public Lumen::VolumeInstance
{
public:
    PTVolumeInstance(PTServiceLocator& a_ServiceLocator);
    PTVolumeInstance(const Lumen::VolumeInstance& a_Instance, PTServiceLocator& a_ServiceLocator);

    void DependencyCallback();
    // Function called under the hood to establish a connection between the instance and the scene
    // Necessary for the acceleration structures to be rebuilt when the transform changes
    void SetSceneRef(PTScene* a_SceneRef);

    void SetVolume(std::shared_ptr<Lumen::ILumenVolume> a_Mesh) override;

    PTScene* m_SceneRef;
    PTServiceLocator& m_Services;

};
