#include "lmnpch.h"
#include "SceneManager.h"
#include "Node.h"
#include "Transform.h"
#include "../Renderer/ILumenMesh.h"

//#include <string>
#include <memory>


Lumen::SceneManager::Scene* Lumen::SceneManager::LoadGLTF(std::string a_Path, glm::mat4& a_TransformMat)
{
	auto findIter = m_LoadedScenes.find(a_Path);

	if(findIter != m_LoadedScenes.end())
	{
		return &(*findIter).second;
	}

	auto& res = m_LoadedScenes[a_Path];		// create new scene at path key
	auto doc = fx::gltf::LoadFromText(a_Path);

	LoadMeshes(doc, res);
	LoadNodes(doc, res, a_TransformMat);
	
	return& res;
}

void Lumen::SceneManager::LoadNodes(fx::gltf::Document& a_Doc, Scene& a_Scene, glm::mat4& a_TransformMat)
{
	//store offsets in the case there is somehow already data loaded in the scene object
	const int nodeOffset = static_cast<int>(a_Scene.m_NodePool.size()) + 1;
	const int meshOffset = static_cast<int>(a_Scene.m_MeshPool.size());
	
	std::vector<std::shared_ptr<Node>> nodes;	//only supporting one scene per file. Seems to work fine 99% of the time

	std::shared_ptr<Node> baseNode = std::make_shared<Node>();
	baseNode->m_NodeID = nodeOffset - 1;
	baseNode->m_LocalTransform = std::make_unique<Transform>(a_TransformMat);
	nodes.push_back(baseNode);
	
	for (auto& fxNodeIdx : a_Doc.scenes.at(0).nodes)
	{
		const fx::gltf::Node& fxNode = a_Doc.nodes.at(fxNodeIdx);
		
		std::shared_ptr<Node> newNode = std::make_shared<Node>();
		newNode->m_MeshID = -1 ? -1 : (fxNode.mesh + meshOffset);
		newNode->m_Name = fxNode.name;
		newNode->m_NodeID = static_cast<int>(nodes.size());

		for (int i = 0; i < static_cast<int>(fxNode.children.size()); i++)
		{
			newNode->m_ChilIndices.push_back(fxNode.children.at(i) + nodeOffset);
		}

		if(fxNode.translation.size() == 3)
		{
			newNode->m_LocalTransform->SetPosition(glm::vec3(
				fxNode.translation[0],
				fxNode.translation[1],
				fxNode.translation[2]
			));
		}

		if (fxNode.rotation.size() == 4)
		{
			newNode->m_LocalTransform->SetRotation(glm::quat(
				fxNode.rotation[0],
				fxNode.rotation[1],
				fxNode.rotation[2],
				fxNode.rotation[3]
			));
		}

		if (fxNode.scale.size() == 3)
		{
			newNode->m_LocalTransform->SetScale(glm::vec3(
				fxNode.scale[0],
				fxNode.scale[1],
				fxNode.scale[2]
			));
		}
		
		a_Scene.m_NodePool.push_back(newNode);
	}
}

void Lumen::SceneManager::LoadMeshes(fx::gltf::Document& a_Doc, Scene& a_Scene, glm::mat4& a_TransformMat)
{
	const int meshOffset = static_cast<int>(a_Scene.m_MeshPool.size());

	std::vector<std::shared_ptr<ILumenMesh>> meshes;

	// pass binary data into mesh


	/*for (auto& fxMesh : a_Doc)
	{

	}*/
	
	//a_Scene.m_MeshPool = 
	
}


