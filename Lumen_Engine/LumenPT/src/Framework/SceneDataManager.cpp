#include "SceneDataManager.h"
//
//#include "../CUDAKernels/SceneDataKernels.cuh"
//
//SceneDataManager::SceneDataManager()
//{
//	//Set up the gpu data struct.
//	CreateGPUSceneData(m_GPUSceneDataBuffer, 1000);
//}
//
//std::unique_ptr<Lumen::ILumenPrimitive> SceneDataManager::LoadPrimitive(const MeshUploadData& a_UploadData)
//{
//	//TODO: Upload the data to the GPU as DeviceMesh. Then create a bottom-level acceleration structure in Optix.
//	//TODO: Then create a new Primitive derived type that contains the DeviceMesh pointer.
//
//	DeviceMesh* mesh;//The GPU struct for meshes.
//
//	return nullptr;
//}
//
//bool SceneDataManager::DeletePrimitive(const std::unique_ptr<Lumen::ILumenPrimitive>& a_Primitive)
//{
//	//TODO: remove the primitive from GPU.
//	return false;
//}
//
//std::uint32_t SceneDataManager::CreateInstance(const InstanceUploadData& a_UploadData)
//{
//	//TODO: Create instance of the given data. Make unique ID, give to Optix and add instance to Optix.
//	//TODO: Create a DeviceInstanceData struct on the GPU that can be looked up with the generated ID.
//	DeviceInstanceData* data;	//The gpu instance struct.
//	return 0;
//}
//
//bool SceneDataManager::UpdateInstance(const std::uint32_t a_InstanceId, const InstanceUploadData& a_Data)
//{
//	//TODO Update an existing instance. Get DeviceInstanceData from the ID and then overwrite old data.
//	DeviceInstanceData* data;	//The gpu instance struct.
//	return false;
//}
//
//bool SceneDataManager::GetInstanceData(const std::uint32_t a_InstanceId, InstanceUploadData& a_Data)
//{
//	//TODO get the instance data for the id (DeviceInstanceData). Then create some upload data struct on the CPU with the right data.
//	return false;
//}
//
//bool SceneDataManager::DeleteInstance(const std::uint32_t a_InstanceId)
//{
//	//TODO Delete an instance for the ID. This deletes the DeviceInstanceData.
//	return false;
//}
