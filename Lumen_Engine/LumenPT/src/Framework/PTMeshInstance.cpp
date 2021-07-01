#include "PTMeshInstance.h"
#include "MemoryBuffer.h"
#include "AccelerationStructure.h"
#include "PTMaterial.h"
#include "PTScene.h"
#include "PTPrimitive.h"
#include "PTServiceLocator.h"
#include "WaveFrontRenderer.h"
#include "SceneDataTable.h"

PTMeshInstance::PTMeshInstance(PTServiceLocator& a_ServiceLocator)
    : m_Services(a_ServiceLocator)
{
    // Register the instance to the dependency callback of its transform.
    // This ensures that DependencyCallback() is called when the transform changes.
    m_Transform.AddDependent(*this);
}


PTMeshInstance::PTMeshInstance(const Lumen::MeshInstance& a_Instance, PTServiceLocator& a_ServiceLocator)
    : m_Services(a_ServiceLocator)
{
    m_Transform = a_Instance.m_Transform;
    m_Transform.AddDependent(*this);
    m_MeshRef = a_Instance.GetMesh();
}

void PTMeshInstance::SetSceneRef(PTScene* a_SceneRef)
{
    // This is called when the mesh is first added to the scene. Essentially immediately flags the scene for an update.
    m_SceneRef = a_SceneRef;
    m_SceneRef->MarkSceneForUpdate();
    MarkSceneDataAsDirty();
}

void PTMeshInstance::DependencyCallback()
{
    m_SceneRef->MarkSceneForUpdate();
    MarkSceneDataAsDirty();
}


void PTMeshInstance::SetMesh(std::shared_ptr<Lumen::ILumenMesh> a_Mesh)
{
    MeshInstance::SetMesh(a_Mesh);
    // Because the mesh used by the instance was changed, the scene's structure needs to be rebuild to reflect the change.
    m_SceneRef->MarkSceneForUpdate();
    MarkSceneDataAsDirty();
}

bool PTMeshInstance::VerifyAccelerationStructure()
{
    // Iterate through all the SDT entries and ensure that the instance IDs used by the acceleration structure are still correct
    bool structCorrect = true;
    for (auto& entry : m_EntryMap)
    {
        auto ptPrimitive = entry.first;
        if (entry.second.m_TableIndex != m_LastUsedPrimitiveIDs[ptPrimitive])
        {
            structCorrect = false;
        }
    }

    // If the struct was found to be incorrect, update it automatically
    if (!structCorrect)
    {
        UpdateAccelerationStructure();
    }

    // Return true if the struct was correct to begin with, return false if it had to be updated
    return structCorrect;
}

void PTMeshInstance::UpdateAccelerationStructure()
{
    std::vector<OptixInstance> instances;

    for (auto& entry : m_EntryMap)
    {
        auto ptPrim = static_cast<const PTPrimitive*>(entry.first);
        // Create the instance data used for the acceleration structure creation
        auto& inst = instances.emplace_back();
        // The instance ID is what is actually used to determine the material and buffer data for the geometry when an intersection occurs
        inst.instanceId = entry.second.m_TableIndex;
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
        
        // Record the instance id used for this primitive in the acceleration structure
        m_LastUsedPrimitiveIDs[entry.first] = entry.second.m_TableIndex;
    }

    // Finally, create the instance acceleration structure out of the instances
    m_AccelerationStructure = m_Services.m_OptixWrapper->BuildInstanceAccelerationStructure(instances);
}

OptixTraversableHandle PTMeshInstance::GetAccelerationStructureHandle() const
{
    return m_AccelerationStructure->m_TraversableHandle;
}

void PTMeshInstance::SetEmissiveness(const Emissiveness& a_EmissiveProperties)
{
    MeshInstance::SetEmissiveness(a_EmissiveProperties);
    MarkSceneDataAsDirty();

}

void PTMeshInstance::SetAdditionalColor(glm::vec4 a_AdditionalColor)
{
    m_AdditionalColor = a_AdditionalColor;
    //UpdateRaytracingData(); // I hate this so much but IDK how to do it better
}

bool PTMeshInstance::UpdateRaytracingData()
{
    if (!m_SceneDataDirty || !m_MeshRef || !m_SceneRef)
        return false; //not updated.

    m_SceneDataDirty = false;

    std::vector<OptixInstance> instances;

    for (auto& prim : m_MeshRef->m_Primitives)
    {
        auto ptPrim = static_cast<PTPrimitive*>(prim.get());

        SceneDataTableEntry<DevicePrimitiveInstance>* entry;
        if (m_EntryMap.find(prim.get()) == m_EntryMap.end())
            m_EntryMap[prim.get()] = m_SceneRef->AddDataTableEntry<DevicePrimitiveInstance>();

        entry = &m_EntryMap.at(prim.get());

        // Fill out the data of the entry here
        // Any instance specific data would go here
        auto& entryData = entry->GetData();

        auto glmTransform = m_Transform.GetTransformationMatrix();
        glmTransform = glm::transpose((glmTransform));

        entryData.m_Primitive = ptPrim->m_DevicePrimitive;
        entryData.m_Transform = sutil::Matrix4x4(reinterpret_cast<float*>(&glmTransform[0]));

        assert(static_cast<int>(m_EmissiveProperties.m_EmissionMode) >= 0 && static_cast<int>(m_EmissiveProperties.m_EmissionMode) < 3);
    	
        entryData.m_EmissionMode = m_EmissiveProperties.m_EmissionMode;
        entryData.m_EmissiveColorAndScale = make_float4(
            m_EmissiveProperties.m_OverrideRadiance.x,
            m_EmissiveProperties.m_OverrideRadiance.y,
            m_EmissiveProperties.m_OverrideRadiance.z,
            m_EmissiveProperties.m_Scale);

        //Check for overridden materials. If defined, overwrite the material pointer.
        if(m_OverrideMaterial != nullptr)
        {
            entryData.m_Primitive.m_Material = std::static_pointer_cast<PTMaterial>(m_OverrideMaterial)->GetDeviceMaterial();
        }

        // The acceleration structure of the mesh instance does not need to be rebuild after this.
        // This is because the scene data table and the acceleration structure are only connected by the entry Ids,
        // which have not changed here

        // It is however necessary to initialize the last used id for this primitive to a value which is almost certain to be invalid 
        m_LastUsedPrimitiveIDs[prim.get()] = -1;
    }

    return true; //Updated.

}
