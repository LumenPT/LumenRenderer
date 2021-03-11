#pragma once

#include "Utils/VectorView.h"

#include "Lumen/ModelLoading/ILumenScene.h"

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <memory>
#include <string>

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
		std::vector<Lumen::MeshInstance> m_InstancedMeshes;
	};

	LumenRenderer(){};
	virtual ~LumenRenderer() = default;

	virtual std::unique_ptr<Lumen::ILumenPrimitive> CreatePrimitive(PrimitiveData& a_MeshData) = 0;
	virtual std::shared_ptr<Lumen::ILumenMesh> CreateMesh(std::vector<std::unique_ptr<Lumen::ILumenPrimitive>>& a_Primitives) = 0;
	virtual std::shared_ptr<Lumen::ILumenTexture> CreateTexture(void* a_PixelData, uint32_t a_Width, uint32_t a_Height) = 0;
	virtual std::shared_ptr<Lumen::ILumenMaterial> CreateMaterial(const MaterialData& a_MaterialData) = 0;
	virtual std::shared_ptr<Lumen::ILumenScene> CreateScene(SceneData a_SceneData);
	virtual std::shared_ptr<Lumen::ILumenVolume> CreateVolume(const std::string& a_FilePath) = 0;
	
	std::shared_ptr<Lumen::ILumenScene> m_Scene;

private:
		
};