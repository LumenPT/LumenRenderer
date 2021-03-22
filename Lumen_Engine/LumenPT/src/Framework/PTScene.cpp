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
    for (Lumen::MeshInstance& mesh : a_SceneData.m_InstancedMeshes)
    {
        auto ptMesh = std::make_unique<PTMeshInstance>(mesh, m_Services);
        ptMesh->SetSceneRef(this);
        m_MeshInstances.push_back(std::move(ptMesh));
    }
}

Lumen::MeshInstance* PTScene::AddMesh()
{
    auto ptMeshInst = std::make_unique<PTMeshInstance>(m_Services);
    ptMeshInst->SetSceneRef(this);
    m_MeshInstances.push_back(std::move(ptMeshInst));
    return m_MeshInstances.back().get();
}

Lumen::VolumeInstance* PTScene::AddVolume()
{
    auto ptVolInst = std::make_unique<PTVolumeInstance>(m_Services);
    ptVolInst->m_SceneRef = this;
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
    for (auto& meshInstance : m_MeshInstances)
    {
        if (meshInstance->GetMesh())
        {
            sbtMatchStructs &= static_cast<PTMesh*>(meshInstance->GetMesh().get())->VerifyStructCorrectness();
        }
    }

    // If there has been a mismatch between the SBT and the acceleration structs, some of the structs have been rebuilt and thus have new handles
    // which invalidates them in the scene struct
    if (m_AccelerationStructureDirty || sbtMatchStructs)
    {
        uint32_t instanceID = 0;
        std::vector<OptixInstance> instances;

        for (auto& meshInstance : m_MeshInstances)
        {
            if (!meshInstance->GetMesh())
                continue;
            auto& ptmi = static_cast<PTMeshInstance&>(*meshInstance);
            auto ptMesh = static_cast<PTMesh*>(meshInstance->GetMesh().get());

            auto& inst = instances.emplace_back();
            inst.traversableHandle = ptMesh->m_AccelerationStructure->m_TraversableHandle;
            inst.sbtOffset = 0;
            inst.visibilityMask = 128;
            inst.instanceId = instanceID++;
            inst.flags = OPTIX_INSTANCE_FLAG_NONE;

            auto transformMat = glm::transpose(ptmi.m_Transform.GetTransformationMatrix());
            memcpy(inst.transform, &transformMat, sizeof(inst.transform));
        }

        for (auto& volumeInstance : m_VolumeInstances)
        {
            if (!volumeInstance->GetVolume())
                continue;
            auto& ptvi = static_cast<PTVolumeInstance&>(*volumeInstance);
            auto ptVolume = static_cast<PTVolume*>(ptvi.GetVolume().get());

            auto& inst = instances.emplace_back();
            inst.traversableHandle = ptVolume->m_AccelerationStructure->m_TraversableHandle;
            inst.sbtOffset = ptVolume->m_RecordHandle.m_TableIndex;
            inst.visibilityMask = 64;
            inst.instanceId = ptVolume->m_SceneEntry.m_TableIndex;
            inst.flags = OPTIX_INSTANCE_FLAG_NONE;

            auto transformMat = glm::transpose(ptvi.m_Transform.GetTransformationMatrix());
            memcpy(inst.transform, &transformMat, sizeof(inst.transform));
        }

        m_AccelerationStructureDirty = false;
#ifdef WAVEFRONT
        m_SceneAccelerationStructure = m_Services.m_OptixWrapper->BuildInstanceAccelerationStructure(instances);
#else
        m_SceneAccelerationStructure = m_Services.m_Renderer->BuildInstanceAccelerationStructure(instances);
#endif
    }

}

