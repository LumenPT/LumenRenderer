#pragma once

#include "gltf.h"
#include "glm/mat4x4.hpp"
#include "../Renderer/LumenRenderer.h"


#include <map>
#include <string>
#include <memory>
#include <vector>

#include "Lumen/Renderer/ILumenResources.h"
#include "VolumeManager.h"

namespace Lumen
{
	class Transform;
	class ILumenPrimitive;
	class ILumenMesh;
	class ILumenScene;
	class Node;
	class ILumenMaterial;
}

namespace Lumen
{
	//Information about a loaded image.
	struct LoadedImageInformation
	{
		std::uint8_t* data = nullptr;
		int w = 0;
		int h = 0;
		int channels = 0;
	};

	//Data about an image in a GLTF file. Could be binary or external.
	class ImageData
	{
	public:
		struct ImageInfo
		{
			std::string FileName{};

			uint32_t BinarySize{};
			uint8_t const* BinaryData{};

			bool IsBinary() const noexcept
			{
				return BinaryData != nullptr;
			}
		};

		explicit ImageData(std::string const& texture)
		{
			m_info.FileName = texture;
		}

		ImageData(fx::gltf::Document const& doc, std::size_t textureIndex, std::string const& modelPath)
		{
			fx::gltf::Image const& image = doc.images[doc.textures[textureIndex].source];

			const bool isEmbedded = image.IsEmbeddedResource();
			if (!image.uri.empty() && !isEmbedded)
			{
				m_info.FileName = fx::gltf::detail::GetDocumentRootPath(modelPath) + "/" + image.uri;
			}
			else
			{
				if (isEmbedded)
				{
					image.MaterializeData(m_embeddedData);
					m_info.BinaryData = &m_embeddedData[0];
					m_info.BinarySize = static_cast<uint32_t>(m_embeddedData.size());
				}
				else
				{
					fx::gltf::BufferView const& bufferView = doc.bufferViews[image.bufferView];
					fx::gltf::Buffer const& buffer = doc.buffers[bufferView.buffer];

					m_info.BinaryData = &buffer.data[bufferView.byteOffset];
					m_info.BinarySize = bufferView.byteLength;
				}
			}
		}

		ImageInfo const& Info() const noexcept
		{
			return m_info;
		}

	private:
		ImageInfo m_info{};

		std::vector<uint8_t> m_embeddedData{};
	};
	 

    // Essentially model loader and manager
	class SceneManager
	{
	public:

		// Note: Scenes can be individual models
		struct GLTFResource	//Deserialized scene data
		{
			std::string											m_Path;
			std::vector<int>									m_RootNodeIndices;
			std::vector<std::shared_ptr<Node>>					m_NodePool;
			std::vector<std::shared_ptr<Lumen::ILumenMesh>>		m_MeshPool;
			std::vector<std::shared_ptr<ILumenMaterial>>		m_MaterialPool;
			std::vector<std::shared_ptr<ILumenScene>>			m_Scenes;
			// Also contain materials
				//binary material data is kinda unsigned long long
			// Also contain textures
			// Also contain lights
			// Also contain camera
			// Also contain sky??  

			// Generic resource. Perhaps contains volumes, too?
		};

		SceneManager() {};
		~SceneManager();

		SceneManager(SceneManager&) = delete;
		SceneManager(SceneManager&&) = delete;
		SceneManager& operator=(SceneManager&) = delete;
		SceneManager& operator=(SceneManager&&) = delete;

		/*
		 * Note: Provide filename and path separately.
		 * Example path: /models/
		 * Example file: duck.gltf
		 */
		GLTFResource* LoadGLTF(std::string a_FileName, std::string a_Path, const glm::mat4& a_TransformMat = glm::mat4(1));	//Load & add to loadedScenes

		void SetPipeline(LumenRenderer& a_Renderer);

		// Removes unused assets from GPU memory. An asset is unused if the no scene or mesh makes use of it.
		void ClearUnusedAssets();

		//TODO: public for testing, make this private later
		VolumeManager m_VolumeManager;

		// Load OpenVDB?

		//Temporary for debugging
		//std::map<std::string, GLTFResource>* GetResourceMap() { &m_LoadedScenes; };

	private:
		std::map<std::string, GLTFResource> m_LoadedScenes;

		//std::vector<std::shared_ptr<Texture>> LoadTextures(fx::gltf::Document& a_Doc, std::string a_Filepath);
		//std::vector<std::shared_ptr<Texture>> LoadMaterials(fx::gltf::Document& a_Doc, GLTFResource& a_resource);
		//std::vector<std::shared_ptr<Texture>> LoadMeshes(fx::gltf::Document& a_Doc, std::string a_Filepath);

		//std::vector<std::shared_ptr<GLTFResource>> LoadScenes(fx::gltf::Document& a_Doc, std::string a_Filepath);

		// Initializes all default resources which are used as a fall back when properties are not found in the file.
		void InitializeDefaultResources();

		void LoadNodes(fx::gltf::Document& a_Doc, GLTFResource& a_Res, int a_NodeId, bool a_Root, const glm::mat4& a_TransformMat = glm::mat4(1));
		void LoadMeshes(fx::gltf::Document& a_Doc, GLTFResource& a_Res); // Load all meshes in the file
		void LoadScenes(fx::gltf::Document& a_Doc, GLTFResource& a_Res); // Load all scenes in the file
		// Load the node with the specified node ID together with all of its children
		void LoadNodeAndChildren(fx::gltf::Document a_Doc, GLTFResource a_Res, ILumenScene& a_Scene, uint32_t a_NodeID, Transform a_ParentTransform = Transform());
		Transform LoadNodeTransform(fx::gltf::Node a_Node); // Load the transform of a node

		//Load a texture from the texture ID.
		LoadedImageInformation LoadTexture(fx::gltf::Document& a_File, int a_TextureId, const std::string& a_Path, int a_NumChannels);

		// Load all materials in the file
		void LoadMaterials(fx::gltf::Document& a_Doc, GLTFResource& a_Res, const std::string& a_Path);
		// Output a vector of binary data from the given accessor index
		std::vector<uint8_t> LoadBinary(fx::gltf::Document& a_Doc, uint32_t a_AccessorIndx);
		uint32_t GetComponentCount(fx::gltf::Accessor& a_Accessor); // Return how many components the accessor uses
		uint32_t GetComponentSize(fx::gltf::Accessor& a_Accessor); // Return the size of the components used by the accessor

		LumenRenderer* m_RenderPipeline;

		// List of all meshes and materials that are currently in use by the renderer
		std::vector<std::shared_ptr<Lumen::ILumenMesh>>		m_InUseMeshes;
		std::vector<std::shared_ptr<ILumenMaterial>>		m_InUseMaterials;

		// Default white texture to use in scenarios when the material's diffuse texture is not specified
		std::shared_ptr<ILumenTexture> m_DefaultDiffuseTexture;

		//Default metallic roughness texture in case the metallic roughness is not specified.
		std::shared_ptr<ILumenTexture> m_DefaultMetalRoughnessTexture;

		//The default normal texture in case none is specified.
		std::shared_ptr<ILumenTexture> m_DefaultNormalTexture;
	};


}