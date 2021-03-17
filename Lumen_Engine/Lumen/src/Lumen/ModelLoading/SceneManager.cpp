#include "lmnpch.h"
#include "SceneManager.h"
#include "Node.h"
#include "Transform.h"
#include "../Renderer/ILumenResources.h"
#include "stb_image.h"
//#include "../../LumenPT/src/Framework/OptiXRenderer.h"

#include <glm/gtc/type_ptr.hpp>

//#include <string>
#include <memory>

Lumen::SceneManager::~SceneManager()
{
    for (auto loadedScene : m_LoadedScenes)
    {
		// Release the meshes and materials from each GLTFResource manually
		auto& res = loadedScene.second;

        for (auto& mesh : res.m_MeshPool)
			mesh.reset();

        for (auto& material : res.m_MaterialPool)
			material.reset();
    }

	// Release the meshes and materials which are no longer parts of GLTFResources
	// due to being in use by the renderer

    for (auto& inUseMesh : m_InUseMeshes)
		inUseMesh.reset();

    for (auto& inUseMaterial : m_InUseMaterials)
		inUseMaterial.reset();
}

Lumen::SceneManager::GLTFResource* Lumen::SceneManager::LoadGLTF(std::string a_FileName, std::string a_Path, const glm::mat4& a_TransformMat)
{
	const auto fullPath = a_Path + a_FileName;
	auto findIter = m_LoadedScenes.find(fullPath);

	if (findIter != m_LoadedScenes.end())
	{
		return &(*findIter).second;
	}

	auto& res = m_LoadedScenes[fullPath];		// create new scene at path key

	//Check for glb or gltf
	const std::string binarySuffix = ".glb";
	bool isBinary = (a_FileName.length() >= binarySuffix.length()) && (0 == a_FileName.compare(a_FileName.length() - binarySuffix.length(), binarySuffix.length(), binarySuffix));

	//NOTE: No quotas specified and no check for .gltf suffix. Might fail to load with large files and wrongly specified suffix.
	fx::gltf::Document doc;
	if(!isBinary)
	{
		doc = fx::gltf::LoadFromText(fullPath);
	}
	else
	{
		doc = fx::gltf::LoadFromBinary((fullPath));
	}

	res.m_Path = fullPath;


	LoadMaterials(doc, res, a_Path);

	LoadMeshes(doc, res);

	//Loop over the root nodes in every scene in this GLTF file, and then add them as instances.
	for (int sceneId = 0; sceneId < doc.scenes.size(); ++sceneId)
	{
		auto& scene = doc.scenes[sceneId];
		for (auto& rootNodeId : scene.nodes)
		{
			LoadNodes(doc, res, rootNodeId, true, a_TransformMat);
		}
	}

	return &res;
}

void Lumen::SceneManager::SetPipeline(LumenRenderer& a_Renderer)
{
	m_RenderPipeline = &a_Renderer;
	m_VolumeManager.SetPipeline(a_Renderer);
}

void Lumen::SceneManager::ClearUnusedAssets()
{
    for (auto loadedScene : m_LoadedScenes)
    {
		auto& res = loadedScene.second;

        for (auto& mesh : res.m_MeshPool)
        {
            if (mesh.use_count() > 1) // Are there other shared pointer instances to this mesh?
            {
                // If yes, store the mesh separately from the GLTF resource
				m_InUseMeshes.push_back(mesh);
            }
			else
			{
			    // Otherwise, reset the pointer, thus destroying the mesh
				mesh.reset();
			}
        }

        for (auto& material : res.m_MaterialPool)
        {
			if (material.use_count() > 1) // Are there other shared pointer instances to this material?
			{
				// If yes, store the material separately from the GLTF resource
				m_InUseMaterials.push_back(material);
			}
			else
			{
				// Otherwise, reset the pointer, thus destroying the material
				material.reset();
			}
        }
    }

	// Verify that none of the meshes and materials that are stored separately
    // have been left unused since the last call to this function

	for (auto& mesh : m_InUseMeshes)
	{
		if (mesh.use_count() <= 1) // Are there no other shared pointer instances to this mesh?
		{
			// If there are none, this mesh can be safely deleted.
			mesh.reset();
		}
	}

	for (auto& material : m_InUseMaterials)
	{
		if (material.use_count() <= 1) // Are there no other shared pointer instances to this material?
		{
			// If there are none, this material can be safely deleted.
			material.reset();
		}
	}
}

void Lumen::SceneManager::LoadNodes(fx::gltf::Document& a_Doc, GLTFResource& a_Res, int a_NodeId, bool a_Root, const glm::mat4& a_TransformMat)
{
	const static glm::mat4 IDENTITY = glm::identity<glm::mat4>();

	auto& node = a_Doc.nodes[a_NodeId];
	glm::mat4 transform = glm::make_mat4(&node.matrix[0]);

	//If the matrix is not defined, load from the other settings.
	if(IDENTITY == transform)
	{
		auto translation = glm::vec3(
			node.translation[0],
			node.translation[1],
			node.translation[2]
		);

		auto rotation = glm::quat(
			node.rotation[0],
			node.rotation[1],
			node.rotation[2],
			node.rotation[3]
		);

		auto scale = glm::vec3(
			node.scale[0],
			node.scale[1],
			node.scale[2]
		);

		Transform t;
		t.SetPosition(translation);
		t.SetRotation(rotation);
		t.SetScale(scale);

		transform = t.GetTransformationMatrix();
	}

	const glm::mat4 chainedTransform = a_TransformMat * transform;

	if(a_Root)
	{
        a_Res.m_RootNodeIndices.push_back(a_NodeId);
	}

	std::shared_ptr<Node> newNode = std::make_shared<Node>();
	newNode->m_MeshID = node.mesh;
	newNode->m_Name = node.name;
	newNode->m_NodeID = static_cast<int>(a_NodeId);
	newNode->m_LocalTransform = std::make_unique<Transform>(chainedTransform);

	for (int i = 0; i < static_cast<int>(node.children.size()); i++)
	{
		newNode->m_ChilIndices.push_back(node.children.at(i));
	}

	a_Res.m_NodePool.push_back(newNode);

	//If there is child meshes, recursively call.
	if (!node.children.empty())
	{
		for (auto& id : node.children)
		{
			LoadNodes(a_Doc, a_Res, id, false, chainedTransform);
		}
	}

	////store offsets in the case there is somehow already data loaded in the scene object
	//const int nodeOffset = static_cast<int>(a_Res.m_NodePool.size()) + 1;
	//const int meshOffset = static_cast<int>(a_Res.m_MeshPool.size());

	//std::vector<std::shared_ptr<Node>> nodes;	//only supporting one scene per file. Seems to work fine 99% of the time

	//std::shared_ptr<Node> baseNode = std::make_shared<Node>();
	//baseNode->m_NodeID = nodeOffset - 1;
	//baseNode->m_LocalTransform = std::make_unique<Transform>(a_TransformMat);
	//nodes.push_back(baseNode);

	//for (auto& fxNodeIdx : a_Doc.scenes.at(a_SceneId).nodes)
	//{
	//	const fx::gltf::Node& fxNode = a_Doc.nodes.at(fxNodeIdx);

	//	std::shared_ptr<Node> newNode = std::make_shared<Node>();
	//	newNode->m_MeshID = -1 ? -1 : (fxNode.mesh + meshOffset);
	//	newNode->m_Name = fxNode.name;
	//	newNode->m_NodeID = static_cast<int>(nodes.size());
	//	newNode->m_LocalTransform = std::make_unique<Transform>();

	//	for (int i = 0; i < static_cast<int>(fxNode.children.size()); i++)
	//	{
	//		newNode->m_ChilIndices.push_back(fxNode.children.at(i) + nodeOffset);
	//	}

	//NOTE: .size() returns the amount of elements (templated) at compile time. It's not the amount of valid values!
	//	if (fxNode.translation.size() == 3)
	//	{
	//		newNode->m_LocalTransform->SetPosition(glm::vec3(
	//			fxNode.translation[0],
	//			fxNode.translation[1],
	//			fxNode.translation[2]
	//		));
	//	}

	//	if (fxNode.rotation.size() == 4)
	//	{
	//		newNode->m_LocalTransform->SetRotation(glm::quat(
	//			fxNode.rotation[0],
	//			fxNode.rotation[1],
	//			fxNode.rotation[2],
	//			fxNode.rotation[3]
	//		));
	//	}

	//	if (fxNode.scale.size() == 3)
	//	{
	//		newNode->m_LocalTransform->SetScale(glm::vec3(
	//			fxNode.scale[0],
	//			fxNode.scale[1],
	//			fxNode.scale[2]
	//		));
	//	}

	//	a_Res.m_NodePool.push_back(newNode);
	//}
}

void Lumen::SceneManager::LoadMeshes(fx::gltf::Document& a_Doc, GLTFResource& a_Res)
{
	const int meshOffset = static_cast<int>(a_Res.m_MeshPool.size());

	std::vector<std::shared_ptr<ILumenMesh>> meshes;
	// pass binary data into mesh

	for (auto& fxMesh : a_Doc.meshes)
	{
		std::vector<std::unique_ptr<Lumen::ILumenPrimitive>> primitives;
		for (auto& fxPrim : fxMesh.primitives)
		{
			std::vector<uint8_t> posBinary, texBinary, norBinary;
			for (auto& fxAttribute : fxPrim.attributes)
			{
				if (fxAttribute.first == "POSITION")
				{
					// store position accessor
					//fxAttribute.second

					posBinary = LoadBinary(a_Doc, fxAttribute.second);
					//make vertex buffer of this, make function

				}
				else if (fxAttribute.first == "TEXCOORD_0")
				{
					texBinary = LoadBinary(a_Doc, fxAttribute.second);
				}
				else if (fxAttribute.first == "COLOR_0")
				{

				}
				else if (fxAttribute.first == "NORMAL")
				{
					norBinary = LoadBinary(a_Doc, fxAttribute.second);
				}
			}
			auto& acc = fxPrim.attributes["POSITION"];
			auto& accessprYe = a_Doc.accessors[acc];
			auto& bufferView = a_Doc.bufferViews[accessprYe.bufferView];

			auto binary = a_Doc.buffers[bufferView.buffer];

			//index to accessor


			auto indexBufferAcc = a_Doc.accessors[fxPrim.indices];
			auto indexBin = LoadBinary(a_Doc, fxPrim.indices);
			auto indexSize = GetComponentSize(indexBufferAcc); // indices are are always a single component
			assert(indexSize <= 4);

			LumenRenderer::PrimitiveData primitiveData;
			primitiveData.m_Positions = posBinary;
			primitiveData.m_TexCoords = texBinary;
			primitiveData.m_Normals = norBinary;

			primitiveData.m_IndexBinary = indexBin;
			primitiveData.m_IndexSize = indexSize;

			primitiveData.m_Material = a_Res.m_MaterialPool[fxPrim.material];

			auto newPrim = m_RenderPipeline->CreatePrimitive(primitiveData);

			//newPrim->m_Material = a_Res.m_MaterialPool[fxPrim.material];
			primitives.push_back(std::move(newPrim));
		}
		meshes.push_back(m_RenderPipeline->CreateMesh(primitives));
	}

	a_Res.m_MeshPool = meshes;

}

Lumen::LoadedImageInformation Lumen::SceneManager::LoadTexture(fx::gltf::Document& a_File, int a_TextureId,
    const std::string& a_Path, int a_NumChannels)
{
	assert(a_TextureId >= 0);

	std::uint8_t* imgData = nullptr;
	int w = 0;
	int h = 0;
	int channels = 0;


	ImageData data(a_File, a_TextureId, a_Path);
	auto info = data.Info();

	stbi_set_flip_vertically_on_load(false);
	if (info.IsBinary())
	{
		//Load from raw
		imgData = stbi_load_from_memory(info.BinaryData, info.BinarySize, &w, &h, &channels, a_NumChannels);
	}
	else
	{
		//Load from file.
		imgData = stbi_load(info.FileName.c_str(), &w, &h, &channels, a_NumChannels);
	}
	assert(imgData != nullptr && "Could not load and decode image for some reason.");

	//If a specific number of channels was requested, overwrite the "would have been" channels count.
	if (a_NumChannels != 0)
	{
		channels = a_NumChannels;
	}
	return LoadedImageInformation{ imgData, w, h, channels };
}

std::vector<uint8_t> Lumen::SceneManager::LoadBinary(fx::gltf::Document& a_Doc, uint32_t a_AccessorIndx)
{
	std::vector<unsigned char> data;
	// Load raw data at accessor index
	auto bufferAccessor = a_Doc.accessors[a_AccessorIndx];
	auto bufferView = a_Doc.bufferViews[bufferAccessor.bufferView];
	auto buffer = a_Doc.buffers[bufferView.buffer];

	uint32_t compCount = GetComponentCount(bufferAccessor);
	uint32_t compSize = GetComponentSize(bufferAccessor);
	uint32_t attSize = compCount * compSize;

	auto stride = std::max(attSize, bufferView.byteStride);

	data.resize(bufferView.byteLength / stride * attSize);

	auto bufferOffset = 0;
	auto gltfBufferOffset = bufferView.byteOffset + bufferAccessor.byteOffset;

	for (uint32_t i = 0; i < bufferAccessor.count; i++)
	{
		memcpy(data.data() + bufferOffset, buffer.data.data() + gltfBufferOffset + bufferOffset, attSize);
		bufferOffset += stride;
	}

	return data;
}

void Lumen::SceneManager::LoadMaterials(fx::gltf::Document& a_Doc, GLTFResource& a_Res, const std::string& a_Path)
{
	std::vector<std::shared_ptr<ILumenMaterial>> materials;	// Get these iLumenMats from OptiXRenderer
	//needs to be a more specific material implementation
	// with actual roughness values etc. 

	for (auto& fxMat : a_Doc.materials)
	{
		LumenRenderer::MaterialData matData;
		auto& mat = materials.emplace_back(m_RenderPipeline->CreateMaterial(matData));

		//Load the diffuse color if not empty.
		if (!fxMat.pbrMetallicRoughness.baseColorFactor.empty()) {
			auto& arr = fxMat.pbrMetallicRoughness.baseColorFactor;
			mat->SetDiffuseColor(glm::vec4(
				arr[0],
				arr[1],
				arr[2],
				arr[3]
			));
		}

		if (fxMat.pbrMetallicRoughness.baseColorTexture.index != -1)
		{
			//auto& fxTex = a_Doc.images.at(fxMat.pbrMetallicRoughness.baseColorTexture.index);
			//int x, y, c;
			//const auto file = (a_Res.m_Path + "/../" + fxTex.uri);
			//const auto stbTex = stbi_load(file.c_str(), &x, &y, &c, 4);
			//const auto tex = m_RenderPipeline->CreateTexture(stbTex, x, y);
			

			//Load the texture either from file or binary. Then create the engine texture object.
			auto info = LoadTexture(a_Doc, fxMat.pbrMetallicRoughness.baseColorTexture.index, a_Path, 4);

			const auto tex = m_RenderPipeline->CreateTexture(info.data, info.w, info.h);
			mat->SetDiffuseTexture(tex);

			//Free the memory after it's uploaded.
			stbi_image_free(info.data);
		}

		if (fxMat.emissiveFactor != std::array<float, 3>{0.0f, 0.0f, 0.0f})
		{
			mat->SetEmission(glm::vec3(
				fxMat.emissiveFactor.at(0),
				fxMat.emissiveFactor.at(1),
				fxMat.emissiveFactor.at(2)
			));
		}

		if (fxMat.emissiveTexture.index != -1)
		{
			auto info = LoadTexture(a_Doc, fxMat.emissiveTexture.index, a_Path, 4);
			const auto tex = m_RenderPipeline->CreateTexture(info.data, info.w, info.h);
			
			mat->SetEmissiveTexture(tex);

			//LMN_INFO("Emissive texture present");
			
			stbi_image_free(info.data);
		}


		//materials.push_back(mat);
	}

	a_Res.m_MaterialPool = materials;
}

uint32_t Lumen::SceneManager::GetComponentSize(fx::gltf::Accessor& a_Accessor)
{

	switch (a_Accessor.componentType)
	{
	case fx::gltf::Accessor::ComponentType::Byte:
	case fx::gltf::Accessor::ComponentType::UnsignedByte:
		return 1;
		break;
	case fx::gltf::Accessor::ComponentType::Short:
	case fx::gltf::Accessor::ComponentType::UnsignedShort:
		return 2;
		break;
	case fx::gltf::Accessor::ComponentType::Float:
	case fx::gltf::Accessor::ComponentType::UnsignedInt:
		return 4;
		break;
	default:
		LMN_ASSERT(0 && "Failed to load GLTF file");
		return 0;
	}

}

uint32_t Lumen::SceneManager::GetComponentCount(fx::gltf::Accessor& a_Accessor)
{
	switch (a_Accessor.type)
	{
	case fx::gltf::Accessor::Type::Scalar:
		return 1;
		break;
	case fx::gltf::Accessor::Type::Vec2:
		return 2;
		break;
	case fx::gltf::Accessor::Type::Vec3:
		return 3;
		break;
	case fx::gltf::Accessor::Type::Mat2:
		return 4;
		break;
	case fx::gltf::Accessor::Type::Mat3:
		return 9;
		break;
	case fx::gltf::Accessor::Type::Mat4:
		return 16;
		break;
	default:
		LMN_ASSERT(0 && "Failed to load GLTF file");
		return 0;
	}
}


