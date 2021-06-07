#include "VolumeManager.h"

#include "Lumen/Renderer/LumenRenderer.h"

#include <filesystem>

namespace Lumen
{
	void VolumeManager::SetPipeline(LumenRenderer& a_Renderer)
	{
		m_RenderPipeline = &a_Renderer;
	}

	VolumeResource* VolumeManager::LoadVDB(const std::string& a_Path)
	{
		//TODO: read all grids from file

		if (!std::filesystem::exists(a_Path))
		{
			LMN_CORE_ERROR("File for VDB loading not found");
			return nullptr;
		}

		auto iter = m_LoadedVolumes.find(a_Path);
		if (iter != m_LoadedVolumes.end())
		{
			return &(iter->second);
		}
		
		VolumeResource& VolumeResource = m_LoadedVolumes[a_Path];
		assert("Renderer reference in VolumeManager not set" && m_RenderPipeline);
		VolumeResource.m_Volume = m_RenderPipeline->CreateVolume(a_Path);
		
		return &VolumeResource;
	}
}
