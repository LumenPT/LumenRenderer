#include "lmnpch.h"
#include "SceneManager.h"
#include "Node.h"
#include "Transform.h"
#include "../Renderer/ILumenResources.h"
#include "stb_image.h"
//#include "../../LumenPT/src/Framework/OptiXRenderer.h"

//#include <string>
#include <memory>

Lumen::SceneManager::GLTFResource* Lumen::SceneManager::LoadGLTF(std::string a_Path, glm::mat4& a_TransformMat)
{
	auto findIter = m_LoadedScenes.find(a_Path);

	if (findIter != m_LoadedScenes.end())
	{
		return &(*findIter).second;
	}

	auto& res = m_LoadedScenes[a_Path];		// create new scene at path key
	auto doc = fx::gltf::LoadFromText(a_Path);

	res.m_Path = a_Path;


	LoadMaterials(doc, res);

	LoadMeshes(doc, res);
	LoadNodes(doc, res, a_TransformMat);

	return &res;
}

void Lumen::SceneManager::SetPipeline(LumenRenderer& a_Renderer)
{
	m_RenderPipeline = &a_Renderer;
	m_VolumeManager.SetPipeline(a_Renderer);
}

void Lumen::SceneManager::LoadNodes(fx::gltf::Document& a_Doc, GLTFResource& a_Res, glm::mat4& a_TransformMat)
{
	//store offsets in the case there is somehow already data loaded in the scene object
	const int nodeOffset = static_cast<int>(a_Res.m_NodePool.size()) + 1;
	const int meshOffset = static_cast<int>(a_Res.m_MeshPool.size());

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
		newNode->m_LocalTransform = std::make_unique<Transform>();

		for (int i = 0; i < static_cast<int>(fxNode.children.size()); i++)
		{
			newNode->m_ChilIndices.push_back(fxNode.children.at(i) + nodeOffset);
		}

		if (fxNode.translation.size() == 3)
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

		a_Res.m_NodePool.push_back(newNode);
	}
}

void Lumen::SceneManager::LoadMeshes(fx::gltf::Document& a_Doc, GLTFResource& a_Res, glm::mat4& a_TransformMat)
{
	const int meshOffset = static_cast<int>(a_Res.m_MeshPool.size());

	std::vector<std::shared_ptr<ILumenMesh>> meshes;
	// pass binary data into mesh

	for (auto& fxMesh : a_Doc.meshes)
	{
		std::vector<std::unique_ptr<Lumen::ILumenPrimitive>> primitives;
		for (auto& fxPrim : fxMesh.primitives)
		{
			std::vector<uint8_t> posBinary, texBinary;
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

void Lumen::SceneManager::LoadMaterials(fx::gltf::Document& a_Doc, GLTFResource& a_Res)
{
	std::vector<std::shared_ptr<ILumenMaterial>> materials;	// Get these iLumenMats from OptiXRenderer
	//needs to be a more specific material implementation
	// with actual roughness values etc. 

	for (auto& fxMat : a_Doc.materials)
	{
		LumenRenderer::MaterialData matData;
		auto& mat = materials.emplace_back(m_RenderPipeline->CreateMaterial(matData));

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
			auto& fxTex = a_Doc.images.at(fxMat.pbrMetallicRoughness.baseColorTexture.index);
			int x, y, c;
			auto stbTex = stbi_load((a_Res.m_Path + "/../" + fxTex.uri).c_str(), &x, &y, &c, 4);
			auto& tex = m_RenderPipeline->CreateTexture(stbTex, x, y);

			mat->SetDiffuseTexture(tex);
		}

		if (fxMat.emissiveFactor != std::array<float, 3>{0.0f, 0.0f, 0.0f})
		{
			mat->SetEmission(glm::vec3(
				fxMat.emissiveFactor.at(0),
				fxMat.emissiveFactor.at(1),
				fxMat.emissiveFactor.at(2)
			));
		}

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


