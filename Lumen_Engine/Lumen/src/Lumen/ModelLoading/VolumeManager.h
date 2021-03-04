#pragma once

#include <map>
#include <string>
#include <memory>
#include <vector>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/IO.h>
#include "nanovdb/util/CudaDeviceBuffer.h"
#include "Lumen/Renderer/ILumenResources.h"
#include "Lumen/Renderer/LumenRenderer.h"

namespace Lumen
{
	//TODO: have each volume store multiple grids for each frame
	using NanoVDBHandle = nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>;
	
	struct VolumeResource
	{
		std::shared_ptr<Lumen::ILumenVolume> m_Volume;
	};
	
	class VolumeManager
	{
	public:
		
		VolumeManager() = default;
		~VolumeManager() = default;

		VolumeManager(VolumeManager&) = delete;
		VolumeManager(VolumeManager&&) = delete;
		VolumeManager& operator=(VolumeManager&) = delete;
		VolumeManager& operator=(VolumeManager&&) = delete;

		void SetPipeline(LumenRenderer& a_Renderer);
		
		VolumeResource* LoadVDB(const std::string& a_Path);
		
	private:
		std::map<std::string, VolumeResource> m_LoadedVolumes;
		LumenRenderer* m_RenderPipeline;
	};
}
