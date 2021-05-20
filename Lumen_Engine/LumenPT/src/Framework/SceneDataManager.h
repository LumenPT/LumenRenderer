#pragma once
//#include <sutil/Matrix.h>
//
//#include "../Shaders/CppCommon/ModelStructs.h"
//#include "MemoryBuffer.h"
//#include "Platform/Windows/LumenRenderer.h"
//#include "Renderer/LumenRenderer.h"
//
///*
// * CPU data to be uploaded as a mesh.
// */
//struct MeshUploadData
//{
//	VectorView<glm::vec3, uint8_t>	m_Positions;	//Vertex positions
//	VectorView<glm::vec2, uint8_t>	m_TexCoords;	//Texture coordinates
//	VectorView<glm::vec3, uint8_t>	m_Normals;		//Normals
//	VectorView<glm::vec4, uint8_t>	m_Tangents;		//Tangents
//	std::vector<uint32_t>			m_Indices;		//The index buffer
//};
//
///*
// * Data about an instance that can be uploaded to the GPU.
// */
//struct InstanceUploadData
//{
//	std::unique_ptr<Lumen::ILumenPrimitive> m_Primitive;	//The primitive containing the actual geometry data.
//
//	sutil::Matrix4x4	m_Transform;		//The transform in world space.
//	bool				m_EnableEmission;	//True when emissive textures should actually emit light.
//	float3				m_Radiance;			//The radiance multiplier for the R, G and B channel.
//};
//
///*
// * Class that allows objects to be added and removed from the scene.
// */
//class SceneDataManager
//{
//public:
//	SceneDataManager();
//
//	/*
//	 * Upload a mesh to the GPU.
//	 * This takes a struct of CPU data and uploads it to the GPU.
//	 * An ILumenPrimitive instance is then constructed to link to the GPU data.
//	 * This ILumenPrimitive can be used to create instances of a particular mesh.
//	 */
//	std::unique_ptr<Lumen::ILumenPrimitive> LoadPrimitive(const MeshUploadData& a_UploadData);
//
//	/*
//	 * Delete a primitive from the GPU.
//	 */
//	bool DeletePrimitive(const std::unique_ptr<Lumen::ILumenPrimitive>& a_Primitive);
//
//	/*
//	 * Create an instance of the provided primitive in the scene.
//	 *
//	 * The InstanceUploadData provided contains various settings for the instance:
//	 * - The transform provided is the final transform in world space.
//	 * - When a_EnableEmission is set to true, this mesh will emit light.
//	 * - a_Radiance gives the radiance per square unit of surface for this mesh.
//	 * - The radiance is only taken into account when a_IsEmissive is set to true.
//	 *
//	 * Returns the newly created instance's unique ID.
//	 */
//	std::uint32_t CreateInstance(
//		const InstanceUploadData& a_UploadData
//	);
//
//	/*
//	 * Update the data of an existing instance.
//	 * If no instance 
//	 */
//	bool UpdateInstance(const std::uint32_t a_InstanceId, const InstanceUploadData& a_Data);
//
//	/*
//	 * Get the data for the given instance ID.
//	 * If the ID was not valid, false is returned.
//	 * When the instance is valid, the data is stored in a_Data and true is returned.
//	 */
//	bool GetInstanceData(const std::uint32_t a_InstanceId, InstanceUploadData& a_Data);
//
//	/*
//	 * Delete an instance from the scene.
//	 * Returns whether the instance was valid and has been deleted.
//	 */
//	bool DeleteInstance(const std::uint32_t a_InstanceId);
//
//private:
//	MemoryBuffer m_GPUSceneDataBuffer;
//	
//};