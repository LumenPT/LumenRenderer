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
    // Can immediately build the acceleration structure out of the provided primitives
    UpdateAccelerationStructure();
}

void PTMesh::UpdateAccelerationStructure()
{
    // Create a list of all instances within the IAS of the mesh
    // Each primitive is one instance
    std::vector<OptixInstance> instances;
    for (auto& primitive : m_Primitives)
    {
        auto ptPrim = static_cast<PTPrimitive*>(primitive.get());
        auto& inst = instances.emplace_back();
        // The instance ID is what is actually used to determine the material and buffer data for the geometry when an intersection occurs
        inst.instanceId = ptPrim->m_SceneDataTableEntry.m_TableIndex;
        // As convention shaders for non-volumetric geometry are kept in the first slot of the shader binding table
        inst.sbtOffset = 0;
        inst.visibilityMask = 255; // Can be intersected by anything
        // Handle to the acceleration structure instance
        inst.traversableHandle = ptPrim->m_GeometryAccelerationStructure->m_TraversableHandle;
        inst.flags = OPTIX_INSTANCE_FLAG_NONE;

        // Initialize the transform of the instance to an identity matrix because primitives are always untransformed in the meshes
        // memcpy is easiest way to achieve this because OptixInstance::transform is a 3x4 matrix, and not a 4x4 matrix
        auto transform = glm::mat4(1.0f);
        memcpy(&inst.transform[0], &transform, sizeof(inst.transform));

        // Record what instance ID was used for this primitive when constructing the acceleration structure
        // This is used to ensure that the instance IDs in the AS and the scene data table match
        m_LastUsedInstanceIDs[ptPrim] = ptPrim->m_SceneDataTableEntry.m_TableIndex;
    }

    // Finally, create the instance acceleration structure out of the instances
#ifdef WAVEFRONT
    m_AccelerationStructure = m_Services.m_OptixWrapper->BuildInstanceAccelerationStructure(instances);
#else 
    m_AccelerationStructure = m_Services.m_Renderer->BuildInstanceAccelerationStructure(instances);
#endif
}

bool PTMesh::VerifyStructCorrectness()
{
    // Iterate through all the primitives and ensure that the instance IDs used by the acceleration structure are still correct
    bool structCorrect = true;
    for (auto& lumenPrimitive : m_Primitives)
    {
        auto ptPrimitive = static_cast<PTPrimitive*>(lumenPrimitive.get());
        if (ptPrimitive->m_SceneDataTableEntry.m_TableIndex != m_LastUsedInstanceIDs[ptPrimitive])
        {
            structCorrect = false;
        }
    }

    // If the struct was found to be incorrect, update it automatically
    if (!structCorrect)
    {
        UpdateAccelerationStructure();
    }

    // True if the struct was correct to begin with, false if it had to be updated
    return structCorrect;
}
