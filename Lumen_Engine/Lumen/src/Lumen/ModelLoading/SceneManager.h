#pragma once

#include "gltf.h"
#include "glm/mat4x4.hpp"

#include <map>
#include <string>
#include <memory>
#include <vector>

namespace Lumen
{
	class Transform;
	class Mesh;
	class Scene;
	class Node;
}

namespace Lumen
{
	
	class SceneManager
	{
	public:

		// Note: Scenes can be individual models
		struct Scene	//Deserialized scene data
		{
			std::vector<int>						m_RootNodeIndices;
			std::vector<std::shared_ptr<Node>>		m_NodePool;
			std::vector<std::shared_ptr<Mesh>>		m_MeshPool;
			// Also contain materials
			// Also contain textures
			// Also contain lights
			// Also contain camera
			// Also contain sky??
			
			// Generic resource. Perhaps contains volumes, too?
		};
		
		SceneManager();
		~SceneManager();

		SceneManager(SceneManager&) = delete;
		SceneManager(SceneManager&&) = delete;
		SceneManager& operator=(SceneManager&) = delete;
		SceneManager& operator=(SceneManager&&) = delete;

		Scene* LoadGLTF(std::string a_Path, glm::mat4& a_TransformMat = glm::mat4(0));	//Load & add to loadedScenes

		// Load OpenVDB?
		
	private:
		std::map<std::string, Scene> m_LoadedScenes;
		
		//std::vector<std::shared_ptr<Texture>> LoadTextures(fx::gltf::Document& a_Doc, std::string a_Filepath);
		//std::vector<std::shared_ptr<Texture>> LoadMaterials(fx::gltf::Document& a_Doc, Scene& a_resource);
		//std::vector<std::shared_ptr<Texture>> LoadMeshes(fx::gltf::Document& a_Doc, std::string a_Filepath);

		//std::vector<std::shared_ptr<Scene>> LoadScenes(fx::gltf::Document& a_Doc, std::string a_Filepath);

		void LoadNodes(fx::gltf::Document& a_Doc, Scene& a_Scene, glm::mat4& a_TransformMat = glm::mat4(0));
		
	};


}
