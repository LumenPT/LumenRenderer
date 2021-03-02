#include "PTVolume.h"

#include <cassert>
#include <filesystem>

#include <nanovdb/util/IO.h>
#include <nanovdb/util/Primitives.h>

#include "MemoryBuffer.h"
#include "AccelerationStructure.h"

PTVolume::PTVolume(PTServiceLocator& a_ServiceLocator)
	: m_Services(a_ServiceLocator)
	
{
}

PTVolume::PTVolume(std::string a_FilePath, PTServiceLocator& a_ServiceLocator)
	: PTVolume(a_ServiceLocator)
{
	//TODO: reinstate
	//Load(a_FilePath);

	//TODO: remove
	m_Handle = nanovdb::createLevelSetSphere<float, nanovdb::CudaDeviceBuffer>(10.0f, nanovdb::Vec3d(0), 0.1);
	//m_Handle = nanovdb::create<float, nanovdb::CudaDeviceBuffer>(10.0f, nanovdb::Vec3d(0), 0.1);
	
	m_Handle.deviceUpload();

	auto* grid = m_Handle.grid<float>();

	auto voxelCount = grid->activeVoxelCount();

	auto gridBBox = grid->indexBBox();
	
}

PTVolume::~PTVolume()
{
	
}

void PTVolume::Load(std::string a_FilePath)
{
	if (!std::filesystem::exists(a_FilePath))
	{
		assert(false); //File for VDB loading not found
	}

	auto filenameExtension = std::filesystem::path(a_FilePath).extension();
	if (filenameExtension == ".vdb")
	{
		assert(false); //Loading of .vdb file not yet supported
	}

	else if (filenameExtension == ".vndb")
	{
		m_Handle = nanovdb::io::readGrid<nanovdb::CudaDeviceBuffer>(a_FilePath);
		m_Handle.deviceUpload();

		
	}

	else
	{
		assert(false); //Filetype not compatible with vdb loading
	}
}
