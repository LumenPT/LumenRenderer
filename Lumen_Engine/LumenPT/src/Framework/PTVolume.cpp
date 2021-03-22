#include "PTVolume.h"

#include <cassert>
#include <filesystem>

#include <nanovdb/util/IO.h>
#include <nanovdb/util/Primitives.h>
#include <nanovdb/util/OpenToNanoVDB.h> 
#include <openvdb/openvdb.h>
#include <openvdb/io/io.h>

#include "MemoryBuffer.h"

PTVolume::PTVolume(PTServiceLocator& a_ServiceLocator)
	: m_Services(a_ServiceLocator)
	
{
}

PTVolume::PTVolume(std::string a_FilePath, PTServiceLocator& a_ServiceLocator)
	: PTVolume(a_ServiceLocator)
{
	//TODO: reinstate
	Load(a_FilePath);

	//TODO: remove
	//m_Handle = nanovdb::createLevelSetSphere<float, nanovdb::CudaDeviceBuffer>(10.0f, nanovdb::Vec3d(0), 0.1);
	//m_Handle = nanovdb::create<float, nanovdb::CudaDeviceBuffer>(10.0f, nanovdb::Vec3d(0), 0.1);
	//m_Handle = nanovdb::createLevelSetTorus<float, nanovdb::CudaDeviceBuffer>(10.0f, 5.0f, nanovdb::Vec3d(0), 0.1);
	//m_Handle = nanovdb::createLevelSetOctahedron<float, nanovdb::CudaDeviceBuffer>(10.0f, nanovdb::Vec3d(0), 0.1);

	//m_Handle.deviceUpload();

	auto* grid = m_Handle.grid<float>();

	auto voxelCount = grid->activeVoxelCount();

	auto gridBBox = grid->indexBBox();
	
}

PTVolume::~PTVolume()
{
	
}

void PTVolume::Load(std::string a_FilePath)
{
	
	openvdb::initialize();
	
	openvdb::FloatGrid::Ptr grid =
		openvdb::FloatGrid::create(/*background value=*/2.0);

	// Associate some metadata with the grid.
	grid->insertMeta("radius", openvdb::FloatMetadata(50.0));
	// Associate a scaling transform with the grid that sets the voxel size
	// to 0.5 units in world space.
	grid->setTransform(
		openvdb::math::Transform::createLinearTransform(/*voxel size=*/0.5));
	// Identify the grid as a level set.
	grid->setGridClass(openvdb::GRID_LEVEL_SET);
	// Name the grid "LevelSetSphere".
	grid->setName("LevelSetSphere");
	
	if (!std::filesystem::exists(a_FilePath))
	{
		assert(false); //File for VDB loading not found
	}

	auto filenameExtension = std::filesystem::path(a_FilePath).extension();
	if (filenameExtension == ".vdb")
	{		
		//assert(false); //Loading of .vdb file not yet supported

		openvdb::io::File file(a_FilePath);

		file.open();

		openvdb::GridBase::Ptr baseGrid;
		for (openvdb::io::File::NameIterator nameIter = file.beginName();
			nameIter != file.endName(); ++nameIter)
		{
			baseGrid = file.readGrid(nameIter.gridName());
			break;
		}
		
		file.close();

		openvdb::FloatGrid::Ptr grid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
		m_Handle = nanovdb::openToNanoVDB<nanovdb::CudaDeviceBuffer>(*grid);
		m_Handle.deviceUpload();
		
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
