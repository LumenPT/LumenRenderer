#include "PTVolume.h"

#include <cassert>
#include <filesystem>

#include <nanovdb/util/IO.h>

#include "MemoryBuffer.h"
#include "AccelerationStructure.h"

PTVolume::PTVolume(PTServiceLocator& a_ServiceLocator)
	: m_Services(a_ServiceLocator)
	
{
}

PTVolume::PTVolume(std::string a_FilePath, PTServiceLocator& a_ServiceLocator)
	: PTVolume(a_ServiceLocator)
{
	Load(a_FilePath);
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
