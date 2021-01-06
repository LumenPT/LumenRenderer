#pragma once

#include "Utils/VectorView.h"

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <memory>

namespace Lumen
{
	class ILumenMesh;
	class ILumenTexture;
	class ILumenMaterial;
}

class LumenRenderer
{
public:

	struct MeshData
	{
		VectorView<glm::vec3, uint8_t> m_Positions;
		VectorView<glm::vec2, uint8_t> m_TexCoords;
		VectorView<glm::vec3, uint8_t> m_Normals;

		// Perhaps temporary solution till we decide how we'll handle the indices
		std::vector<uint8_t> m_IndexBinary;
		size_t m_IndexSize;
	};

	struct MaterialData
	{
		glm::vec4 m_DiffuseColor;
		std::shared_ptr<Lumen::ILumenTexture> m_DiffuseTexture;
		std::shared_ptr<Lumen::ILumenTexture> m_NormalMap;
	};

	LumenRenderer(){};

	virtual std::shared_ptr<Lumen::ILumenMesh> CreateMesh(const MeshData& a_MeshData) = 0;
	virtual std::shared_ptr<Lumen::ILumenTexture> CreateTexture(void* a_PixelData, uint32_t a_Width, uint32_t a_Height) = 0;
	virtual std::shared_ptr<Lumen::ILumenMaterial> CreateMaterial(const MaterialData& a_MaterialData) = 0;

private:
		
};