#pragma once

#include "gltf.h"
#include "glm/mat4x4.hpp"
#include "../Renderer/LumenRenderer.h"


#include <map>
#include <string>
#include <memory>
#include <vector>

namespace Lumen
{
	class Transform;
	class ILumenMesh;
	class Scene;
	class Node;
	class ILumenMaterial;
}

namespace Lumen
{
	
	class SceneManager
	{
	public:

		// Note: Scenes can be individual models
		struct Scene	//Deserialized scene data
		{
			std::string											m_Path;
			std::vector<int>									m_RootNodeIndices;
			std::vector<std::shared_ptr<Node>>					m_NodePool;
			std::vector<std::shared_ptr<Lumen::ILumenMesh>>		m_MeshPool;
			std::vector<std::shared_ptr<ILumenMaterial>>		m_MaterialPool;
			// Also contain materials
				//binary material data is kinda unsigned long long
			// Also contain textures
			// Also contain lights
			// Also contain camera
			// Also contain sky??  
			
			// Generic resource. Perhaps contains volumes, too?
		};
		
		SceneManager() {};
		~SceneManager(){};

		SceneManager(SceneManager&) = delete;
		SceneManager(SceneManager&&) = delete;
		SceneManager& operator=(SceneManager&) = delete;
		SceneManager& operator=(SceneManager&&) = delete;

		Scene* LoadGLTF(std::string a_Path, glm::mat4& a_TransformMat = glm::mat4(0));	//Load & add to loadedScenes

		void SetPipeline(LumenRenderer& a_Renderer);

		// Load OpenVDB?
		
	private:
		std::map<std::string, Scene> m_LoadedScenes;
		
		//std::vector<std::shared_ptr<Texture>> LoadTextures(fx::gltf::Document& a_Doc, std::string a_Filepath);
		//std::vector<std::shared_ptr<Texture>> LoadMaterials(fx::gltf::Document& a_Doc, Scene& a_resource);
		//std::vector<std::shared_ptr<Texture>> LoadMeshes(fx::gltf::Document& a_Doc, std::string a_Filepath);

		//std::vector<std::shared_ptr<Scene>> LoadScenes(fx::gltf::Document& a_Doc, std::string a_Filepath);

		void LoadNodes(fx::gltf::Document& a_Doc, Scene& a_Scene, glm::mat4& a_TransformMat = glm::mat4(0));
		void LoadMeshes(fx::gltf::Document& a_Doc, Scene& a_Scene, glm::mat4& a_TransformMat = glm::mat4(0));
		void LoadMaterials(fx::gltf::Document& a_Doc, Scene& a_Scene);
		std::vector<uint8_t> LoadBinary(fx::gltf::Document& a_Doc, uint32_t a_AccessorIndx);
		uint32_t GetComponentCount(fx::gltf::Accessor& a_Accessor);
		uint32_t GetComponentSize(fx::gltf::Accessor& a_Accessor);
		
		LumenRenderer* m_RenderPipeline;

	};


}
