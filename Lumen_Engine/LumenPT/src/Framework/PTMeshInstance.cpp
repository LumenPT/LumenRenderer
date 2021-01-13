#include "PTMeshInstance.h"

#include "AccelerationStructure.h"

#include "PTScene.h"


#include "OptiXRenderer.h"
#include "PTPrimitive.h"

PTMeshInstance::PTMeshInstance(PTServiceLocator& a_ServiceLocator)
    : m_Services(a_ServiceLocator)
{
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
    m_SceneRef = a_SceneRef;
    m_SceneRef->AddMeshInstanceForUpdate(*this);
}

void PTMeshInstance::DependencyCallback()
{
    m_SceneRef->AddMeshInstanceForUpdate(*this);
}


void PTMeshInstance::SetMesh(std::shared_ptr<Lumen::ILumenMesh> a_Mesh)
{
    MeshInstance::SetMesh(a_Mesh);
    m_SceneRef->AddMeshInstanceForUpdate(*this);    
}
