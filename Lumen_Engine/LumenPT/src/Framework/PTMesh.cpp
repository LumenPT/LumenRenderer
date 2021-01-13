#include "PTMesh.h"
#include "AccelerationStructure.h"
#include "PTPrimitive.h"
#include "PTServiceLocator.h"
#include "WaveFrontRenderer.h"

#include "glm/mat4x4.hpp"

PTMesh::PTMesh(std::vector<std::unique_ptr<Lumen::ILumenPrimitive>>& a_Primitives, PTServiceLocator& a_ServiceLocator)
    : ILumenMesh(a_Primitives)
    , m_Services(a_ServiceLocator)
{
    std::vector<OptixInstance> instances;
    uint32_t idCounter = 0;
    for (auto& primitive : m_Primitives)
    {
        auto ptPrim = static_cast<PTPrimitive*>(primitive.get());
        auto& inst = instances.emplace_back();
        inst.instanceId = ++idCounter;
        inst.sbtOffset = ptPrim->m_RecordHandle.m_TableIndex;
        inst.visibilityMask = 255;
        inst.traversableHandle = ptPrim->m_GeometryAccelerationStructure->m_TraversableHandle;
        inst.flags = OPTIX_INSTANCE_FLAG_NONE;

        auto transform = glm::mat4(1.0f);
        memcpy(&inst.transform[0], &transform, sizeof(inst.transform));
    }

    m_AccelerationStructure = m_Services.m_Renderer->BuildInstanceAccelerationStructure(instances);
}
