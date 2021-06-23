#pragma once

#include "Utils/VectorView.h"

#include "Lumen/ModelLoading/ILumenScene.h"

#include "Lumen/ModelLoading/SceneManager.h"

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <memory>
#include <mutex>
#include <string>
#include <filesystem>
#include "Glad/glad.h"

class FrameSnapshot;
class Camera;

namespace Lumen
{
	class ILumenPrimitive;
	class ILumenTexture;
	class ILumenMaterial;
}

struct FrameStats
{
	uint64_t m_Id = 0;
	std::map<std::string, uint64_t> m_Times;
	// Expand if necessary
};

// Base class for the renderer which is used to abstract away API implementation details
class LumenRenderer
{
public:

	

	// Structu used to initialize a primitive
	struct PrimitiveData
	{
		PrimitiveData() : m_Interleaved(false) {}
		bool m_Interleaved;

		VectorView<glm::vec3, uint8_t> m_Positions;
		VectorView<glm::vec2, uint8_t> m_TexCoords;
		VectorView<glm::vec3, uint8_t> m_Normals;
		VectorView<glm::vec4, uint8_t> m_Tangents;

		std::vector<uint8_t> m_VertexBinary;

		// Perhaps temporary solution till we decide how we'll handle the indices
		std::vector<uint8_t> m_IndexBinary;
		size_t m_IndexSize;

		std::shared_ptr<Lumen::ILumenMaterial> m_Material;
	};

	// Struct used to initialize a material
	struct MaterialData
	{
		MaterialData(): m_DiffuseColor(1.f, 1.f, 1.f, 1.f), m_EmissionVal(0.f, 0.f, 0.f), m_TransmissionFactor(0.f),
                        m_ClearCoatFactor(0),
                        m_ClearCoatRoughnessFactor(0),
                        m_IndexOfRefraction(1.f),
                        m_SpecularFactor(0),
                        m_SpecularTintFactor(0),
						m_SubSurfaceFactor(0),
                        m_Luminance(1.f),
                        m_Anisotropic(0),
                        m_SheenFactor(0),
						m_SheenTintFactor(0),
						m_MetallicFactor(1.f),
						m_RoughnessFactor(1.f),
                        m_TintFactor(1.f, 1.f, 1.f),
                        m_Transmittance(1.f, 1.f, 1.f)
        {
        }

		glm::vec4 m_DiffuseColor;
		glm::vec3 m_EmissionVal;
		std::shared_ptr<Lumen::ILumenTexture> m_DiffuseTexture;
		std::shared_ptr<Lumen::ILumenTexture> m_NormalMap;
		std::shared_ptr<Lumen::ILumenTexture> m_MetallicRoughnessTexture;
		std::shared_ptr<Lumen::ILumenTexture> m_EmissiveTexture;

		//Disney!
		std::shared_ptr<Lumen::ILumenTexture> m_TransmissionTexture;
		std::shared_ptr<Lumen::ILumenTexture> m_ClearCoatTexture;
		std::shared_ptr<Lumen::ILumenTexture> m_ClearCoatRoughnessTexture;
		std::shared_ptr<Lumen::ILumenTexture> m_TintTexture;

		float m_TransmissionFactor;
		float m_ClearCoatFactor;
		float m_ClearCoatRoughnessFactor;
		float m_IndexOfRefraction;
		float m_SpecularFactor;
		float m_SpecularTintFactor;
		float m_SubSurfaceFactor;
		float m_Luminance;
		float m_Anisotropic;
		float m_SheenFactor;
		float m_SheenTintFactor;
		float m_MetallicFactor;
		float m_RoughnessFactor;
		glm::vec3 m_TintFactor;
		glm::vec3 m_Transmittance;
	};

	// Struct used to initialize a scene
	struct SceneData
	{
		glm::vec3 m_CameraPosition = glm::vec3(0, 0, 0);
		glm::vec3 m_CameraUp = glm::vec3(0, 1, 0);
		glm::mat4 m_CameraTrans = glm::mat4(1);
		//Camera m_Camera;

		std::vector<Lumen::MeshInstance> m_InstancedMeshes;
	};

	// Struct used to initialize the renderer
	struct InitializationData
	{

		std::filesystem::path m_AssetDirectory;
		std::filesystem::path m_ShaderDirectory;

		uint8_t m_MaxDepth;
		uint8_t m_RaysPerPixel;
		uint8_t m_ShadowRaysPerPixel;

		//uint2 m_RenderResolution;
		//uint2 m_OutputResolution;
		glm::vec2 m_RenderResolution;
		glm::vec2 m_OutputResolution;

	};

	LumenRenderer()	{};
	LumenRenderer(const InitializationData& a_InitializationData) {};
	virtual ~LumenRenderer() = default;

	// The following functions are abstract because they are used to hide away implementation details connected to the rendering APIs.
	// Essentially all of them need to be implemented by the child classes to match the API requirements.

	// Create a primitive from the provided primitive data
	virtual void StartRendering() = 0;
	virtual void PerformDeferredOperations() {};

	virtual Lumen::SceneManager::GLTFResource OpenCustomFileFormat(const std::string& a_OriginalFilePath);
	virtual Lumen::SceneManager::GLTFResource CreateCustomFileFormat(const std::string& a_OriginalFilePath);

	virtual std::unique_ptr<Lumen::ILumenPrimitive> CreatePrimitive(PrimitiveData& a_MeshData) = 0;
	// Create a mesh from the provided primitives
	virtual std::shared_ptr<Lumen::ILumenMesh> CreateMesh(std::vector<std::shared_ptr<Lumen::ILumenPrimitive>>& a_Primitives) = 0;
	// Create a texture from the provided texture data
	virtual std::shared_ptr<Lumen::ILumenTexture> CreateTexture(void* a_PixelData, uint32_t a_Width, uint32_t a_Height, bool a_Normalize) = 0;
	// Create a material from the provided material data
	std::shared_ptr<Lumen::ILumenMaterial> CreateMaterial();
	virtual std::shared_ptr<Lumen::ILumenMaterial> CreateMaterial(const MaterialData& a_MaterialData) = 0;
	// Create a scene from the provided scene data
	virtual std::shared_ptr<Lumen::ILumenScene> CreateScene(SceneData a_SceneData = {});
	// Create a volume from the provided file path
	virtual std::shared_ptr<Lumen::ILumenVolume> CreateVolume(const std::string& a_FilePath) = 0;

	//NGX test
	virtual void InitNGX() = 0;

	void CreateDefaultResources();

	virtual unsigned int GetOutputTexture() = 0;	//scene argument may be redundant... or not
	virtual std::vector<uint8_t> GetOutputTexturePixels(uint32_t& a_Width, uint32_t& a_Height) = 0;

	virtual void SetRenderResolution(glm::uvec2 a_NewResolution) = 0;
	virtual void SetOutputResolution(glm::uvec2 a_NewResolution) = 0;
	/*
	 * Set the blend mode.
	 * When set to true, output is blended instead of overwritten.
	 */
	virtual void SetBlendMode(bool a_Blend) = 0;

	void SetRenderResolution(uint32_t a_NewWidth, uint32_t a_NewHeight) { SetRenderResolution(glm::uvec2(a_NewWidth, a_NewHeight)); }
	void SetOutputResolution(uint32_t a_NewWidth, uint32_t a_NewHeight) { SetOutputResolution(glm::uvec2(a_NewWidth, a_NewHeight)); }

	virtual glm::uvec2 GetRenderResolution() = 0;
	virtual glm::uvec2 GetOutputResolution() = 0;

	/*
	 * Get the blend mode.
	 * When true, output is blended and not overwritten.
	 */
	virtual bool GetBlendMode() const = 0;

	virtual void BeginSnapshot() = 0;

	virtual std::unique_ptr<FrameSnapshot> EndSnapshot() = 0;
	std::shared_ptr<Lumen::ILumenScene> m_Scene;

	FrameStats GetLastFrameStats();

	//Debug GLuint texture accessible by application
	GLuint m_DebugTexture;

	unsigned int m_DlssMode = 2; //corresponds to dlssmodes enum in DLSS init params
protected:

	FrameStats m_LastFrameStats;
	std::mutex m_FrameStatsMutex;

private:
	std::shared_ptr<Lumen::ILumenTexture> m_DefaultWhiteTexture;
	std::shared_ptr<Lumen::ILumenTexture> m_DefaultNormalTexture;
	std::shared_ptr<Lumen::ILumenTexture> m_DefaultDiffuseTexture;

};