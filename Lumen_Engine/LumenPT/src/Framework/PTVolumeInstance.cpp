#include "PTVolumeInstance.h"
#include "MemoryBuffer.h"
#include "AccelerationStructure.h"
#include "PTScene.h"
#include "PTPrimitive.h"
#include "SceneDataTable.h"
#include "PTVolume.h"

PTVolumeInstance::PTVolumeInstance(PTServiceLocator& a_ServiceLocator)
    : m_Services(a_ServiceLocator)
{
    m_Transform.AddDependent(*this);
}

PTVolumeInstance::PTVolumeInstance(const Lumen::VolumeInstance& a_Instance, PTServiceLocator& a_ServiceLocator)
    : m_Services(a_ServiceLocator)
{
    m_Transform = a_Instance.m_Transform;
    m_Transform.AddDependent(*this);
    m_VolumeRef = a_Instance.GetVolume();
}

void PTVolumeInstance::DependencyCallback()
{
    m_SceneRef->MarkSceneForUpdate();
	UpdateRaytracingData();
}

void PTVolumeInstance::SetSceneRef(PTScene* a_SceneRef)
{
    m_SceneRef = a_SceneRef;
    m_SceneRef->MarkSceneForUpdate();
	m_SceneDataTableEntry = m_SceneRef->AddDataTableEntry<DeviceVolume>();
	UpdateRaytracingData();
}

void PTVolumeInstance::SetVolume(std::shared_ptr<Lumen::ILumenVolume> a_Mesh)
{
    VolumeInstance::SetVolume(a_Mesh);
    m_SceneRef->MarkSceneForUpdate();
	UpdateRaytracingData();
}

void PTVolumeInstance::UpdateRaytracingData()
{
	if (!m_VolumeRef || !m_SceneRef)
		return;

	auto& entryData = m_SceneDataTableEntry.GetData();
	auto* ptVolume = reinterpret_cast<PTVolume*>(m_VolumeRef.get());
	entryData.m_Grid = ptVolume->GetHandle()->grid<float>();
}
