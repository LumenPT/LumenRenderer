#pragma once
#include "ModelLoading/MeshInstance.h"

#include "Optix/optix_types.h"

#include "SceneDataTableEntry.h"
#include "../Shaders/CppCommon/ModelStructs.h"


class PTScene;
class PTServiceLocator;
class AccelerationStructure;

// Extension of the base class which takes API implementation details into account
class PTMeshInstance : public Lumen::MeshInstance
{
public:
    // Default constructor creates an instance without a mesh attached to it
    // In this case the instance is not rendered because there is no geometry associated with it
    PTMeshInstance(PTServiceLocator& a_ServiceLocator);
    // Constructor used when building a scene out of given scene data. This is necessary to convert from
    // the base mesh instance class to this one because of the service locator requirement
    PTMeshInstance(const Lumen::MeshInstance& a_Instance, PTServiceLocator& a_ServiceLocator);

    // Function that gets called when the transform is modified externally
    // Used to notify the scene that the instance has been transformed, and an acceleration structure update is in order
    void DependencyCallback();

    // Function called under the hood to establish a connection between the instance and the scene
    // Necessary for the acceleration structures to be rebuilt when the transform changes
    void SetSceneRef(PTScene* a_SceneRef);

    void SetMesh(std::shared_ptr<Lumen::ILumenMesh> a_Mesh) override;

    bool VerifyAccelerationStructure();

    void UpdateAccelerationStructure();

    OptixTraversableHandle GetAccelerationStructureHandle() const;

    PTScene* m_SceneRef; // Reference to the scene the instance is a part of
    PTServiceLocator& m_Services; // Reference to the path tracer service locator

    void SetEmissiveness(const Emissiveness& a_EmissiveProperties) override;

    void SetAdditionalColor(glm::vec4 a_AdditionalColor) override;

    std::unordered_map<Lumen::ILumenPrimitive*, SceneDataTableEntry<DevicePrimitiveInstance>>& GetInstanceEntryMap()
    {
        return m_EntryMap;
    }

    void SetOverrideMaterial(std::shared_ptr<Lumen::ILumenMaterial> a_OverrideMaterial) override
    {
        MeshInstance::SetOverrideMaterial(a_OverrideMaterial);
        MarkSceneDataAsDirty();
    };

    virtual void UpdateAccelRemoveThis() override
    {
        UpdateRaytracingData();
    }

    PTScene* GetSceneRef() { return m_SceneRef; }


    bool UpdateRaytracingData();
private:
    void MarkSceneDataAsDirty() { m_SceneDataDirty = true; }
private:

    std::unique_ptr<AccelerationStructure> m_AccelerationStructure;
    std::unordered_map<Lumen::ILumenPrimitive*, SceneDataTableEntry<DevicePrimitiveInstance>> m_EntryMap;
    bool m_SceneDataDirty;
    std::unordered_map<Lumen::ILumenPrimitive*, uint32_t> m_LastUsedPrimitiveIDs;

};
