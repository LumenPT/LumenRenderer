#include "PTScene.h"
#include "AccelerationStructure.h"

#include "PTMesh.h"

#include "PTMeshInstance.h"

#include "WaveFrontRenderer.h"

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

void PTScene::AddMeshInstanceForUpdate(PTMeshInstance& a_Handle)
{
    m_TransformedAccelerationStructures.emplace(&a_Handle);
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
        sbtMatchStructs &= static_cast<PTMesh*>(meshInstance->GetMesh().get())->VerifyStructCorrectness();
    }

    // If there has been a mismatch between the SBT and the acceleration structs, some of the structs have been rebuilt and thus have new handles
    // which invalidates them in the scene struct
    if (!m_TransformedAccelerationStructures.empty() || sbtMatchStructs)
    {
        uint32_t instanceID = 0;
        std::vector<OptixInstance> instances;

        for (auto& meshInstance : m_MeshInstances)
        {
            auto& ptmi = static_cast<PTMeshInstance&>(*meshInstance);
            auto ptMesh = static_cast<PTMesh*>(meshInstance->GetMesh().get());

            auto transformMat = glm::transpose(ptmi.m_Transform.GetTransformationMatrix());
            auto& inst = instances.emplace_back();
            inst.traversableHandle = ptMesh->m_AccelerationStructure->m_TraversableHandle;
            inst.sbtOffset = 0;
            inst.visibilityMask = 255;
            inst.instanceId = ++instanceID;
            inst.flags = OPTIX_INSTANCE_FLAG_NONE;

            memcpy(inst.transform, &transformMat, sizeof(inst.transform));
        }
        m_TransformedAccelerationStructures.clear();
        m_SceneAccelerationStructure = m_Services.m_Renderer->BuildInstanceAccelerationStructure(instances);
    }

}
