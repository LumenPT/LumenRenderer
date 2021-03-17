#pragma once

#include "Utils/VectorView.h"

#include "Lumen/ModelLoading/ILumenScene.h"

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <memory>
#include <string>


class Camera;

namespace Lumen
{
	class ILumenPrimitive;
	class ILumenTexture;
	class ILumenMaterial;
}

class LumenRenderer
{
public:

	struct PrimitiveData
	{
		VectorView<glm::vec3, uint8_t> m_Positions;
		VectorView<glm::vec2, uint8_t> m_TexCoords;
		VectorView<glm::vec3, uint8_t> m_Normals;

		VectorView<glm::vec3, uint8_t> m_VertexBuffer; //interleaved data. May not be needed here.
		// Perhaps temporary solution till we decide how we'll handle the indices
		std::vector<uint8_t> m_IndexBinary;
		size_t m_IndexSize;

		std::shared_ptr<Lumen::ILumenMaterial> m_Material;
	};

	struct MaterialData
	{
		glm::vec4 m_DiffuseColor;
		glm::vec4 m_EmssivionVal;
		std::shared_ptr<Lumen::ILumenTexture> m_DiffuseTexture;
		std::shared_ptr<Lumen::ILumenTexture> m_NormalMap;
	};

	struct SceneData
	{
		glm::vec3 m_CameraPosition = glm::vec3(0,0,0);
		glm::vec3 m_CameraUp = glm::vec3(0,1,0);
		glm::mat4 m_CameraTrans = glm::mat4(1);
		//Camera m_Camera;
		
		std::vector<Lumen::MeshInstance> m_InstancedMeshes;
	};

	struct InitializationData
	{

		uint8_t m_MaxDepth;
		uint8_t m_RaysPerPixel;
		uint8_t m_ShadowRaysPerPixel;

		//uint2 m_RenderResolution;
		//uint2 m_OutputResolution;
		glm::vec2 m_RenderResolution;
		glm::vec2 m_OutputResolution;

	};

	LumenRenderer(){};
	LumenRenderer(const InitializationData& a_InitializationData){};
	virtual ~LumenRenderer() = default;

	virtual std::unique_ptr<Lumen::ILumenPrimitive> CreatePrimitive(PrimitiveData& a_MeshData) = 0;
	virtual std::shared_ptr<Lumen::ILumenMesh> CreateMesh(std::vector<std::unique_ptr<Lumen::ILumenPrimitive>>& a_Primitives) = 0;
	virtual std::shared_ptr<Lumen::ILumenTexture> CreateTexture(void* a_PixelData, uint32_t a_Width, uint32_t a_Height) = 0;
	virtual std::shared_ptr<Lumen::ILumenMaterial> CreateMaterial(const MaterialData& a_MaterialData) = 0;
	virtual std::shared_ptr<Lumen::ILumenScene> CreateScene(SceneData a_SceneData);
	virtual std::shared_ptr<Lumen::ILumenVolume> CreateVolume(const std::string& a_FilePath) = 0;

	virtual unsigned int TraceFrame(std::shared_ptr<Lumen::ILumenScene>& a_Scene) = 0;	//scene argument may be redundant... or not 

	std::shared_ptr<Lumen::ILumenScene> m_Scene;

private:
		
};