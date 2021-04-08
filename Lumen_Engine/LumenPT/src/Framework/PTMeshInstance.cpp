#include "PTMeshInstance.h"
#include "MemoryBuffer.h"
#include "AccelerationStructure.h"
#include "PTScene.h"
#include "PTPrimitive.h"
#include "PTServiceLocator.h"

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
}

void PTMeshInstance::DependencyCallback()
{
    m_SceneRef->MarkSceneForUpdate();
}


void PTMeshInstance::SetMesh(std::shared_ptr<Lumen::ILumenMesh> a_Mesh)
{
    MeshInstance::SetMesh(a_Mesh);
    // Because the mesh used by the instance was changed, the scene's structure needs to be rebuild to reflect the change.
    m_SceneRef->MarkSceneForUpdate();    
}
