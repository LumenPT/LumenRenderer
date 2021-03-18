#include "PTMesh.h"
#include "AccelerationStructure.h"
#include "PTPrimitive.h"
#include "PTServiceLocator.h"
#include "RendererDefinition.h"
#include "OptixWrapper.h"

#include "glm/mat4x4.hpp"

PTMesh::PTMesh(std::vector<std::unique_ptr<Lumen::ILumenPrimitive>>& a_Primitives, PTServiceLocator& a_ServiceLocator)
    : ILumenMesh(a_Primitives)
    , m_Services(a_ServiceLocator)
{
    UpdateAccelerationStructure();
}

void PTMesh::UpdateAccelerationStructure()
{
    std::vector<OptixInstance> instances;
    uint32_t idCounter = 0;
    for (auto& primitive : m_Primitives)
    {
        auto ptPrim = static_cast<PTPrimitive*>(primitive.get());
        auto& inst = instances.emplace_back();
        inst.instanceId = ptPrim->m_SceneDataTableEntry.m_TableIndex;
        inst.sbtOffset = 0;
        inst.visibilityMask = 255;
        inst.traversableHandle = ptPrim->m_GeometryAccelerationStructure->m_TraversableHandle;
        inst.flags = OPTIX_INSTANCE_FLAG_NONE;

        auto transform = glm::mat4(1.0f);
        memcpy(&inst.transform[0], &transform, sizeof(inst.transform));
        m_LastUsedInstanceIDs[ptPrim] = ptPrim->m_SceneDataTableEntry.m_TableIndex;
    }

    
#ifdef WAVEFRONT
    m_AccelerationStructure = m_Services.m_OptixWrapper->BuildInstanceAccelerationStructure(instances);
#else 
    m_AccelerationStructure = m_Services.m_Renderer->BuildInstanceAccelerationStructure(instances);
#endif
}

bool PTMesh::VerifyStructCorrectness()
{
    bool structCorrect = true;
    for (auto& lumenPrimitive : m_Primitives)
    {
        auto ptPrimitive = static_cast<PTPrimitive*>(lumenPrimitive.get());
        if (ptPrimitive->m_SceneDataTableEntry.m_TableIndex != m_LastUsedInstanceIDs[ptPrimitive])
        {
            structCorrect = false;
        }
    }

    if (!structCorrect)
    {
        UpdateAccelerationStructure();
    }

    // True if the struct was correct to begin with, false if it had to be updated
    return structCorrect;
}
