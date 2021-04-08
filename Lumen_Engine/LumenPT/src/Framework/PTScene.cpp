#include "PTScene.h"
#include "AccelerationStructure.h"
#include "PTMesh.h"
#include "PTMeshInstance.h"
#include "PTServiceLocator.h"
#include "PTVolume.h"
#include "PTVolumeInstance.h"
#include "RendererDefinition.h"

PTScene::PTScene(LumenRenderer::SceneData& a_SceneData, PTServiceLocator& a_ServiceLocator)
    : m_Services(a_ServiceLocator)
{
    // Iterates through all mesh instances in the provided scene data struct
    // and converts them to the path tracer specific instance type
    for (Lumen::MeshInstance& mesh : a_SceneData.m_InstancedMeshes)
    {
        auto ptMesh = std::make_unique<PTMeshInstance>(mesh, m_Services);
        ptMesh->SetSceneRef(this);
        m_MeshInstances.push_back(std::move(ptMesh));
    }
}

Lumen::MeshInstance* PTScene::AddMesh()
{
    // Create the new mesh of the path tracer mesh instance type
    auto ptMeshInst = std::make_unique<PTMeshInstance>(m_Services);
    // Set the scene reference for the new instance
    ptMeshInst->SetSceneRef(this);
    // Add the instance to the list of all instances and return a raw pointer to it
    m_MeshInstances.push_back(std::move(ptMeshInst));
    return m_MeshInstances.back().get();
}

Lumen::VolumeInstance* PTScene::AddVolume()
{
    // Create the new mesh of the path tracer volume instance type
    auto ptVolInst = std::make_unique<PTVolumeInstance>(m_Services);
    // Set the scene reference for the new instance
    ptVolInst->m_SceneRef = this;
    // Add the instance to the list of all instances and return a raw pointer to it
    m_VolumeInstances.push_back(std::move(ptVolInst));
    return m_VolumeInstances.back().get();
}

void PTScene::Clear()
{
    // Do the same as the parent class
    // and mark the acceleration structure as dirty and waiting for an update
    ILumenScene::Clear();
    m_AccelerationStructureDirty = true;
}

void PTScene::MarkSceneForUpdate()
{
    m_AccelerationStructureDirty = true;
}

OptixTraversableHandle PTScene::GetSceneAccelerationStructure()
{
    UpdateSceneAccelerationStructure();

    return m_SceneAccelerationStructure->m_TraversableHandle;
}

void PTScene::UpdateSceneAccelerationStructure()
{
    bool sbtMatchStructs = true;
    // Go through all mesh instances, and verify that the meshes used by them are still correct
    for (auto& meshInstance : m_MeshInstances)
    {
        if (meshInstance->GetMesh())
        {
            // VerifyStructCorrectness returns false if the acceleration structure was rebuilt.            
            sbtMatchStructs &= static_cast<PTMesh*>(meshInstance->GetMesh().get())->VerifyStructCorrectness();
        }
    }

    // If there has been a mismatch between the SBT and the acceleration structs, some of the structs have been rebuilt and thus have new handles
    // which invalidates them in the scene struct
    // This is most likely to happen if resources were freed from the GPU, which would trigger a scene data table rebuild.
    // This rebuild is unlikely to mark the scene acceleration struct as dirty, but verifying that the mesh acceleration structures are still correct
    // will pick up on that
    if (m_AccelerationStructureDirty || !sbtMatchStructs)
    {
        // Create a list of all acceleration structure instances that need to be in the scene acceleration structure
        std::vector<OptixInstance> instances;

        // First go through the static geometry structures
        for (auto& meshInstance : m_MeshInstances)
        {
            if (!meshInstance->GetMesh())
                continue;
            auto& ptmi = static_cast<PTMeshInstance&>(*meshInstance);
            auto ptMesh = static_cast<PTMesh*>(meshInstance->GetMesh().get());

            auto& inst = instances.emplace_back();
            // Record the acceleration structure of the mesh into the OptixInstance
            inst.traversableHandle = ptMesh->m_AccelerationStructure->m_TraversableHandle;
            inst.sbtOffset = 0; // Optix states that if the AS instance is an IAS, the sbt offset must be 0
            inst.visibilityMask = 0x80; // 128
            // The instance ID of this struct is irrelevant, as the intersections will see the ID of the lowest level AS
            inst.instanceId = 0; 
            inst.flags = OPTIX_INSTANCE_FLAG_NONE;

            // Get the transformation matrix from the transform of the instance
            // We need to transpose it, because GLM and Optix use different matrix types
            // If we do not transpose the matrix, the translation would be lost completely
            auto transformMat = glm::transpose(ptmi.m_Transform.GetTransformationMatrix());
            // Copy the transformation matrix into the OptixInstance struct
            memcpy(inst.transform, &transformMat, sizeof(inst.transform));
        }

        // Then go through all volume instances in the scene
        for (auto& volumeInstance : m_VolumeInstances)
        {
            if (!volumeInstance->GetVolume())
                continue;
            auto& ptvi = static_cast<PTVolumeInstance&>(*volumeInstance);
            auto ptVolume = static_cast<PTVolume*>(ptvi.GetVolume().get());

            auto& inst = instances.emplace_back();
            // Record the acceleration structure of the volume into the OptixInstance
            inst.traversableHandle = ptVolume->m_AccelerationStructure->m_TraversableHandle;
            // TODO: This needs to be changed to represent the index of the volumetrics shaders in the shader binding table
            inst.sbtOffset = ptVolume->m_RecordHandle.m_TableIndex; 
            inst.visibilityMask = 0x40; // 64
            // For volumetrics, the instance ID is important because it determine which volumetric data will be displayed
            inst.instanceId = ptVolume->m_SceneEntry.m_TableIndex;
            inst.flags = OPTIX_INSTANCE_FLAG_NONE;

            // Get the transformation matrix from the transform of the instance
            // We need to transpose it, because GLM and Optix use different matrix types
            // If we do not transpose the matrix, the translation would be lost completely
            auto transformMat = glm::transpose(ptvi.m_Transform.GetTransformationMatrix());
            // Copy the transformation matrix into the OptixInstance struct
            memcpy(inst.transform, &transformMat, sizeof(inst.transform));
        }

        // We rebuild the acceleration structure from scratch because it is cheap to do so,
        // and that way the quality of the structure does not drop when instances in it are moved
#ifdef WAVEFRONT
        m_SceneAccelerationStructure = m_Services.m_OptixWrapper->BuildInstanceAccelerationStructure(instances);
#else
        m_SceneAccelerationStructure = m_Services.m_Renderer->BuildInstanceAccelerationStructure(instances);
#endif
        // The scene acceleration structure is no longer dirty 
        m_AccelerationStructureDirty = false;
    }

}

