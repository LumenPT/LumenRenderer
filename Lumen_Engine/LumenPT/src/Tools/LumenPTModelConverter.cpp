#include "LumenPTModelConverter.h"

#include "Utils/VectorView.h"
//#include "../../../../LumenPT/src/Shaders/CppCommon/ModelStructs.h" // I hate this as much as whoever is reading this
#include "Lumen/Core.h"
#include "AssetLoading/AssetLoading.h"
#include "../Shaders/CppCommon/ModelStructs.h"

#include <glm/geometric.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <fstream>
#include <filesystem>
#include <glm/gtc/type_ptr.hpp>



#include "../Framework/PTTexture.h"
#include "Renderer/LumenRenderer.h"

namespace fs = std::filesystem;
using namespace fx::gltf;


Lumen::SceneManager::GLTFResource LumenPTModelConverter::ConvertGLTF(std::string a_SourcePath)
{
	auto workDir = fs::current_path();
	auto p = (a_SourcePath);

	fs::path sp(p);

	fx::gltf::Document fxDoc;

	fx::gltf::ReadQuotas readQuotas{};
	readQuotas.MaxBufferCount = 99;
	readQuotas.MaxBufferByteLength = 999999000000;
	readQuotas.MaxFileSize = 999999000000;

	printf("[File Conversion] Starting to convert %s to Ollad.\n", a_SourcePath.c_str());

	if (sp.extension() == ".gltf")
		fxDoc = LoadFromText(p, readQuotas);
	else if (sp.extension() == ".glb")
		fxDoc = LoadFromBinary(p, readQuotas);
	
	printf("[File Conversion] Done loading file from GLTF.\n");

	auto content = GenerateContent(fxDoc, p);

	printf("[File Conversion] Done generating content.\n");
	
	auto header = GenerateHeader(content);

	printf("[File Conversion] Done generating header.\n");

	volatile auto dbgfc = &content;

	std::string destPath = std::filesystem::path(p).replace_extension(ms_ExtensionName).string();

	OutputToFile(header, content.m_Blob, destPath);

	printf("[File Conversion] Done outputting to file.\n");

	content.m_Textures.clear();

	return LoadFile(destPath);
}

Lumen::SceneManager::GLTFResource LumenPTModelConverter::LoadFile(std::string a_SourcePath)
{
	Lumen::SceneManager::GLTFResource res;
	res.m_Path = a_SourcePath;

	std::ifstream ifs(a_SourcePath, std::ios::in | std::ios::binary | std::ios::ate);

	if (ifs.is_open()) // Was the file opened successfully?
	{
		// If yes, start decoding the data in it

	    // Get the full size of the file
		ifs.seekg(0, ifs.end);
		uint64_t fileSize = ifs.tellg();
		ifs.seekg(0, ifs.beg);

		// Read the first bytes which point to the size of the header
		decltype(Header::m_Size) headerSize;
		ifs.read(reinterpret_cast<char*>(&headerSize), sizeof(headerSize));

		// Set the cursor to the end of the header
		ifs.seekg(headerSize, ifs.cur);

		// Copy all of the binary payload used by the header into a buffer
		auto dataSize = fileSize - sizeof(headerSize) - headerSize;
		std::vector<char> dataBin(dataSize);
		ifs.read(dataBin.data(), dataSize);

		// Move the cursor back to the beginning of the header, right after the number signifying the size of the header
		ifs.seekg(sizeof(headerSize), ifs.beg);

		uint64_t numTex;
		ifs.read(reinterpret_cast<char*>(&numTex), sizeof(uint64_t));

		std::vector<std::shared_ptr<Lumen::ILumenTexture>> textures;

		for (size_t i = 0; i < numTex; i++)
		{
			HeaderTexture ht;
			ifs.read(reinterpret_cast<char*>(&ht), sizeof(ht));

			int32_t x, y, c;

			// TODO: This seems to be the actual bottleneck in the loading process.
			// Could maybe move the image decompression to the conversion process.
			auto texData = stbi_load_from_memory(reinterpret_cast<uint8_t*>(&dataBin[ht.m_Offset]), ht.m_Size, &x, &y, &c, 4);

			std::vector<uint8_t> arr(x * y);
			memcpy(arr.data(), texData, x * y);

            if (ht.m_TextureType == TextureType::EMetalRoughness)
            {
				for (int index = 0; index < x * y; ++index)
				{
					uchar4* data = reinterpret_cast<uchar4*>(&texData[index * 4]);
					data->y = std::max(data->y, static_cast<unsigned char>(1));	//Minimal value for roughness is 1/255 (as a char).
				}
            }

			bool normalize = ht.m_TextureType == TextureType::EDiffuse || ht.m_TextureType == TextureType::EEmissive;

			textures.push_back(m_RendererRef->CreateTexture(texData, x, y, normalize));
		}

		uint64_t numMat;
		ifs.read(reinterpret_cast<char*>(&numMat), sizeof(uint64_t));

		for (size_t i = 0; i < numMat; i++)
		{
			HeaderMaterial hm;
			ifs.read(reinterpret_cast<char*>(&hm), sizeof(hm));

			

			LumenRenderer::MaterialData matData;
			matData.m_DiffuseColor.x = hm.m_Color[0];
			matData.m_DiffuseColor.y = hm.m_Color[1];
			matData.m_DiffuseColor.z = hm.m_Color[2];
			matData.m_DiffuseColor.w = hm.m_Color[3];

			matData.m_EmissionVal.x = hm.m_Emission[0];
			matData.m_EmissionVal.y = hm.m_Emission[1];
			matData.m_EmissionVal.z = hm.m_Emission[2];

			if (hm.m_DiffuseTextureId != -1)
				matData.m_DiffuseTexture = textures[hm.m_DiffuseTextureId];
			else
				matData.m_DiffuseTexture = m_DefaultWhiteTexture;

			if (hm.m_NormalMapId != -1)
				matData.m_NormalMap = textures[hm.m_NormalMapId];
			else
				matData.m_NormalMap = m_DefaultNormalTexture;

			if (hm.m_MetallicRoughnessTextureId != -1)
				matData.m_MetallicRoughnessTexture = textures[hm.m_MetallicRoughnessTextureId];
			else
			    matData.m_MetallicRoughnessTexture = m_DefaultMetalRoughnessTexture;

            if (hm.m_EmissiveTextureId != -1)
			    matData.m_EmissiveTexture = textures[hm.m_EmissiveTextureId];
			else
				matData.m_EmissiveTexture = m_DefaultEmissiveTexture;


			//Load Disney stuff
			matData.m_TransmissionTexture = hm.m_TransmissionTextureId != -1 ? textures[hm.m_TransmissionTextureId] : m_DefaultWhiteTexture;
			matData.m_ClearCoatTexture = hm.m_ClearCoatTextureId != -1 ? textures[hm.m_ClearCoatTextureId] : m_DefaultWhiteTexture;
			matData.m_ClearCoatRoughnessTexture = hm.m_ClearCoatRoughnessTextureId != -1 ? textures[hm.m_ClearCoatRoughnessTextureId] : m_DefaultWhiteTexture;
			matData.m_TintTexture = hm.m_TintTextureId != -1 ? textures[hm.m_TintTextureId] : m_DefaultWhiteTexture;

			matData.m_TransmissionFactor = hm.m_TransmissionFactor;
			matData.m_ClearCoatFactor = hm.m_ClearCoatFactor;
			matData.m_ClearCoatRoughnessFactor = hm.m_ClearCoatRoughnessFactor;
			matData.m_IndexOfRefraction = hm.m_IndexOfRefraction;
			matData.m_SpecularFactor = hm.m_SpecularFactor;
			matData.m_SpecularTintFactor = hm.m_SpecularTintFactor;
			matData.m_SubSurfaceFactor = hm.m_SubSurfaceFactor;
			matData.m_Luminance = hm.m_Luminance;
			matData.m_Anisotropic = hm.m_Anisotropic;
			matData.m_SheenFactor = hm.m_SheenFactor;
			matData.m_SheenTintFactor = hm.m_SheenTintFactor;
			matData.m_TintFactor = glm::vec3(hm.m_TintFactor[0], hm.m_TintFactor[1], hm.m_TintFactor[2]);
			matData.m_Transmittance = glm::vec3(hm.m_Transmittance[0], hm.m_Transmittance[1], hm.m_Transmittance[2]);

			matData.m_MetallicFactor = hm.m_MetallicFactor;
			matData.m_RoughnessFactor = hm.m_RoughnessFactor;




			res.m_MaterialPool.push_back(m_RendererRef->CreateMaterial(matData));
		}

		uint64_t numMeshes;
		ifs.read(reinterpret_cast<char*>(&numMeshes), sizeof(uint64_t));

		for (size_t i = 0; i < numMeshes; i++)
		{
			decltype(HeaderMesh::m_Header) meshHeader;
			ifs.read(reinterpret_cast<char*>(&meshHeader), sizeof(meshHeader));

			std::vector<std::shared_ptr<Lumen::ILumenPrimitive>> primitives;

			for (size_t j = 0; j < meshHeader.m_NumPrimitives; j++)
			{
				HeaderPrimitive hp;

				ifs.read(reinterpret_cast<char*>(&hp), sizeof(hp));

				LumenRenderer::PrimitiveData primData;
				primData.m_IndexSize = hp.m_IndexSize;
				primData.m_IndexBinary.resize(hp.m_IndexBufferSize);
				memcpy(primData.m_IndexBinary.data(), &dataBin[hp.m_IndexBufferOffset], hp.m_IndexBufferSize);

				primData.m_Interleaved = true;
				primData.m_VertexBinary.resize(hp.m_VertexBufferSize);
				memcpy(primData.m_VertexBinary.data(), &dataBin[hp.m_VertexBufferOffset], hp.m_VertexBufferSize);


				std::vector<Vertex> arr(primData.m_VertexBinary.size() / sizeof(Vertex));

				memcpy(arr.data(), primData.m_VertexBinary.data(), primData.m_VertexBinary.size());

				primData.m_Material = res.m_MaterialPool[hp.m_MaterialId];

				primitives.push_back(m_RendererRef->CreatePrimitive(primData));

			}

			res.m_MeshPool.push_back(m_RendererRef->CreateMesh(primitives));
		}

		uint64_t numScenes;
		ifs.read(reinterpret_cast<char*>(&numScenes), sizeof(numScenes));
		for (size_t i = 0; i < numScenes; i++)
		{
			decltype(HeaderScene::m_Header) sceneHeader;
			ifs.read(reinterpret_cast<char*>(&sceneHeader), sizeof(sceneHeader));



			res.m_Scenes.push_back(m_RendererRef->CreateScene());
			auto& scene = res.m_Scenes.back();

			scene->m_Name.resize(sceneHeader.m_NameLength);
			ifs.read(scene->m_Name.data(), sceneHeader.m_NameLength);

			uint32_t unnamedCounter = 0;

			for (size_t j = 0; j < sceneHeader.m_NumMeshes; j++)
			{
				
				decltype(HeaderMeshInstance::m_Header) instanceHeader;
				ifs.read(reinterpret_cast<char*>(&instanceHeader), sizeof(instanceHeader));
				
				auto m = scene->AddMesh();
				m->SetMesh(res.m_MeshPool[instanceHeader.m_MeshId]);
				m->m_Transform = glm::make_mat4(instanceHeader.m_Transform);
				m->m_Name.resize(instanceHeader.m_NameLength);
				ifs.read(m->m_Name.data(), m->m_Name.size());

                if (m->m_Name.empty())
                {
					m->m_Name = std::string("Mesh ") + std::to_string(unnamedCounter++);
                }

			}
		}
	}
	else
		res.m_Path = ""; // Empty path means no file 

	return res;
}

void LumenPTModelConverter::SetRendererRef(LumenRenderer& a_Renderer)
{
	m_DefaultWhiteTexture.reset();
	m_DefaultMetalRoughnessTexture.reset();
	m_DefaultNormalTexture.reset();
	m_DefaultEmissiveTexture.reset();

	m_RendererRef = &a_Renderer;
	uchar4 whitePixel = { 255,255,255,255 };
	uchar4 diffusePixel{ 255, 255, 255, 255 };
	uchar4 normal = { 128, 128, 255, 0 };
	m_DefaultWhiteTexture = m_RendererRef->CreateTexture(&whitePixel, 1, 1, true);
	m_DefaultMetalRoughnessTexture = m_RendererRef->CreateTexture(&diffusePixel, 1, 1, false);
	m_DefaultNormalTexture = m_RendererRef->CreateTexture(&normal, 1, 1, false);
	m_DefaultEmissiveTexture = m_RendererRef->CreateTexture(&whitePixel, 1, 1, true);
}

LumenPTModelConverter::FileContent LumenPTModelConverter::GenerateContent(const fx::gltf::Document& a_FxDoc, const std::string& a_SourcePath)
{
	FileContent fc;
	
    for (uint32_t i = 0; i < a_FxDoc.images.size(); i++)
    {
		fc.m_Textures.push_back(TextureToBlob(a_FxDoc, i, fc.m_Blob, a_SourcePath));
		printf("[File Conversion] Extracted image %i of %i.\n", i , static_cast<int>(a_FxDoc.images.size()) - 1);
    }

	int index = 0;
    for (auto& material : a_FxDoc.materials)
    {
		auto& m = fc.m_Materials.emplace_back();

		m.m_Color[0] = material.pbrMetallicRoughness.baseColorFactor[0];
		m.m_Color[1] = material.pbrMetallicRoughness.baseColorFactor[1];
		m.m_Color[2] = material.pbrMetallicRoughness.baseColorFactor[2];
		m.m_Color[3] = material.pbrMetallicRoughness.baseColorFactor[3];

		m.m_Emission[0] = material.emissiveFactor[0];
		m.m_Emission[1] = material.emissiveFactor[1];
		m.m_Emission[2] = material.emissiveFactor[2];

		m.m_DiffuseTextureId = material.pbrMetallicRoughness.baseColorTexture.index;
        if (m.m_DiffuseTextureId != -1)
        {
			m.m_DiffuseTextureId = a_FxDoc.textures[m.m_DiffuseTextureId].source;
		    fc.m_Textures[m.m_DiffuseTextureId].m_TextureType = TextureType::EDiffuse;
        }
    	
		assert(m.m_DiffuseTextureId < static_cast<int>(a_FxDoc.textures.size()));

		m.m_NormalMapId = material.normalTexture.index;
		if (m.m_NormalMapId != -1)
		{
			m.m_NormalMapId = a_FxDoc.textures[m.m_NormalMapId].source;
			fc.m_Textures[m.m_NormalMapId].m_TextureType = TextureType::ENormal;
		}

		assert(m.m_NormalMapId < static_cast<int>(a_FxDoc.textures.size()));

		m.m_MetallicRoughnessTextureId = material.pbrMetallicRoughness.metallicRoughnessTexture.index;
		if (m.m_MetallicRoughnessTextureId != -1)
		{
			m.m_MetallicRoughnessTextureId = a_FxDoc.textures[m.m_MetallicRoughnessTextureId].source;
			fc.m_Textures[m.m_MetallicRoughnessTextureId].m_TextureType = TextureType::EMetalRoughness;
		}

		assert(m.m_MetallicRoughnessTextureId < static_cast<int>(a_FxDoc.textures.size()));

		m.m_EmissiveTextureId = material.emissiveTexture.index;
		if (m.m_EmissiveTextureId != -1)
		{
			m.m_EmissiveTextureId = a_FxDoc.textures[m.m_EmissiveTextureId].source;
			fc.m_Textures[m.m_EmissiveTextureId].m_TextureType = TextureType::EEmissive;
		}


		assert(m.m_EmissiveTextureId < static_cast<int>(a_FxDoc.textures.size()));

		//Extract the metal roughness scaling factors.
		m.m_MetallicFactor = material.pbrMetallicRoughness.metallicFactor;
		m.m_RoughnessFactor = std::fmaxf(0.01f, material.pbrMetallicRoughness.roughnessFactor);	//Can't be 0. 1/255 is smallest, so 0.01 falls within that.

		//Add the Disney stuff

		//Non-GLTF specified values.
		m.m_Luminance = 1.f;
		m.m_Transmittance[0] = 0.f;	//Default to 0 because this is actually how much is absorbed.
		m.m_Transmittance[1] = 0.f;
		m.m_Transmittance[2] = 0.f;
		m.m_SubSurfaceFactor = 0.f;
		m.m_Anisotropic = 0.f;

		nlohmann::json extensions;
		if(material.extensionsAndExtras.contains("extensions"))
		{
			extensions = material.extensionsAndExtras["extensions"];
		}
    	
        //Transmission
		constexpr auto transmissionExtension = "KHR_materials_transmission";
		if(extensions.contains(transmissionExtension))
		{
			auto json = extensions[transmissionExtension];
			const float transmissionFactor = JsonGetOrDefault<float>(json, "transmissionFactor", 0.f);
			m.m_TransmissionFactor = transmissionFactor;
			
			const uint32_t transmissionTextureId = JsonGetOrDefault<uint32_t>(json, "transmissionTexture", -1);
			m.m_TransmissionTextureId = transmissionTextureId;

			if (m.m_TransmissionTextureId != -1)
			{
				m.m_TransmissionTextureId = a_FxDoc.textures[m.m_TransmissionTextureId].source;
				fc.m_Textures[m.m_TransmissionTextureId].m_TextureType = TextureType::ETransmissive;
			}
		}
		//Not specified, default values.
		else
		{
			m.m_TransmissionTextureId = -1;
			m.m_TransmissionFactor = 0.f;
		}

		//Sheen
		constexpr auto sheenExtension = "KHR_materials_sheen";
		if (extensions.contains(sheenExtension))
		{
			//Note: Disney and GLTF seem to have quite different sheen properties. Just do this I guess and then manually set if we wanna tweak it.
			auto json = extensions[sheenExtension];
			const float sheenFactor = JsonGetOrDefault<float>(json, "sheenRoughnessFactor", 0.f);
			m.m_SheenFactor = sheenFactor;
			m.m_SheenTintFactor = 1.f;
		}
		//Not specified, default values.
		else
		{
			m.m_SheenFactor = 0.f;
			m.m_SheenTintFactor = 0.f;
		}

		//IOR
		constexpr auto iorExtension = "KHR_materials_ior";
		if (extensions.contains(iorExtension))
		{
			auto json = extensions[iorExtension];
			const float ior = JsonGetOrDefault<float>(json, "ior", 1.5f);
			m.m_IndexOfRefraction = ior;
		}
		//Not specified, default values.
		else
		{
			m.m_IndexOfRefraction = 1.5f;
		}

		//Clearcoat
		constexpr auto clearCoatExtension = "KHR_materials_clearcoat";
		if (extensions.contains(clearCoatExtension))
		{
			auto json = extensions[clearCoatExtension];

			m.m_ClearCoatRoughnessTextureId = JsonGetOrDefault<uint32_t>(json, "clearcoatRoughnessTexture", -1);
			m.m_ClearCoatRoughnessTextureId = a_FxDoc.textures[m.m_ClearCoatRoughnessTextureId].source;
			m.m_ClearCoatTextureId = JsonGetOrDefault<uint32_t>(json, "clearcoatTexture", -1);
			m.m_ClearCoatTextureId = a_FxDoc.textures[m.m_ClearCoatTextureId].source;

			if (m.m_ClearCoatRoughnessTextureId != -1)
			{
				m.m_ClearCoatRoughnessTextureId = a_FxDoc.textures[m.m_ClearCoatRoughnessTextureId].source;
				fc.m_Textures[m.m_ClearCoatRoughnessTextureId].m_TextureType = TextureType::EClearCoatRoughness;
			}

			if (m.m_ClearCoatTextureId != -1)
			{
				m.m_ClearCoatTextureId = a_FxDoc.textures[m.m_ClearCoatTextureId].source;
				fc.m_Textures[m.m_ClearCoatTextureId].m_TextureType = TextureType::EClearCoat;
			}
			
			m.m_ClearCoatRoughnessFactor = JsonGetOrDefault<float>(json, "clearcoatRoughnessFactor", 0.f);
			m.m_ClearCoatFactor = JsonGetOrDefault<float>(json, "clearcoatFactor", 0.f);

		}
		//Not specified, default values.
		else
		{
			m.m_ClearCoatRoughnessTextureId = -1;
			m.m_ClearCoatRoughnessFactor = 0.f;
			m.m_ClearCoatFactor = 0.f;
			m.m_ClearCoatTextureId = -1;
		}

		//Specular and Tint (loaded from the specular stuff I guess). Sheen also uses the tint.
		constexpr auto specularExtension = "KHR_materials_specular";
		if (extensions.contains(specularExtension))
		{
			auto json = extensions[specularExtension];

			m.m_SpecularFactor = JsonGetOrDefault<float>(json, "specularFactor", 0.f);
			m.m_TintTextureId = JsonGetOrDefault<uint32_t>(json, "specularColorTexture", -1);
			m.m_SpecularTintFactor = 1.f;

			if (m.m_TintTextureId != -1)
			{
				m.m_TintTextureId = a_FxDoc.textures[m.m_TintTextureId].source;
				fc.m_Textures[m.m_TintTextureId].m_TextureType = TextureType::ETint;
			}
		}
		//Not specified, default values.
		else
		{
			m.m_SpecularFactor = 0.f;
			m.m_TintTextureId = -1;
			m.m_SpecularTintFactor = 0.f;
		}

		printf("[File Conversion] Done generating material %i of %i.\n", index, static_cast<int>(a_FxDoc.materials.size()) - 1);
		++index;
	}

	index = 0;
	
    for (auto& mesh : a_FxDoc.meshes)
    {
		auto& m = fc.m_Meshes.emplace_back();
        for (auto& primitive : mesh.primitives)
        {
			m.m_Primitives.push_back(PrimitiveToBlob(a_FxDoc, primitive, fc.m_Blob));
        }
		m.m_Header.m_NumPrimitives = m.m_Primitives.size();
		printf("[File Conversion] Done generating mesh %i of %i.\n", index, static_cast<int>(a_FxDoc.meshes.size()) - 1);
		++index;
    }

	index = 0;
    for (auto& fxScene : a_FxDoc.scenes)
    {
		fc.m_Scenes.push_back(MakeScene(a_FxDoc, fxScene));
		printf("[File Conversion] Done extracting scene %i of %i.\n", index, static_cast<int>(a_FxDoc.scenes.size()) - 1);
		++index;
    }

	fc.m_Blob.Trim();

	return fc;
}

LumenPTModelConverter::Header LumenPTModelConverter::GenerateHeader(const FileContent& a_Content)
{
	Header h;

	auto numTex = a_Content.m_Textures.size();
	auto numMat = a_Content.m_Materials.size();
	auto numMesh = a_Content.m_Meshes.size();
	auto numScenes = a_Content.m_Scenes.size();

	h.m_Binary.Write(reinterpret_cast<char*>(&numTex), sizeof(numTex));
	h.m_Binary.Write(a_Content.m_Textures.data(), a_Content.m_Textures.size() * sizeof(HeaderTexture));
	h.m_Binary.Write(reinterpret_cast<char*>(&numMat), sizeof(numMat));
	h.m_Binary.Write(a_Content.m_Materials.data(), a_Content.m_Materials.size() * sizeof(HeaderMaterial));

	h.m_Binary.Write(reinterpret_cast<char*>(&numMesh), sizeof(numMesh));
	for (auto mesh : a_Content.m_Meshes)
    {
		h.m_Binary.Write(&mesh.m_Header, sizeof(mesh.m_Header));
		h.m_Binary.Write(mesh.m_Primitives.data(), mesh.m_Primitives.size() * sizeof(HeaderPrimitive));
    }


	h.m_Binary.Write(reinterpret_cast<char*>(&numScenes), sizeof(numScenes));
    for (auto headerScene : a_Content.m_Scenes)
    {
		h.m_Binary.Write(&headerScene.m_Header, sizeof(headerScene.m_Header));
		h.m_Binary.Write(headerScene.m_Name.data(), headerScene.m_Name.size());
        for (auto& mesh : headerScene.m_Meshes)
        {
			h.m_Binary.Write(&mesh.m_Header, sizeof(mesh.m_Header));
			h.m_Binary.Write(mesh.m_Name.data(), mesh.m_Name.size());
		    //h.m_Binary.Write(headerScene.m_Meshes.data(), headerScene.m_Meshes.size() * sizeof(HeaderMeshInstance));            
        }
    }

	h.m_Binary.Trim();
	h.m_Size = h.m_Binary.m_Size;
	return h;
}

void LumenPTModelConverter::OutputToFile(const Header& a_Header, const Blob& a_Binary, const std::string& a_DestPath)
{
	std::ofstream ofs(a_DestPath, std::ios::out | std::ios::binary);

	ofs.write(reinterpret_cast<const char*>(&a_Header.m_Size), sizeof(a_Header.m_Size));
	ofs.write(a_Header.m_Binary.m_Data.data(), a_Header.m_Binary.m_Size);
	ofs.write(a_Binary.m_Data.data(), a_Binary.m_Size);

	ofs.close();
}

LumenPTModelConverter::HeaderTexture LumenPTModelConverter::TextureToBlob(const fx::gltf::Document& a_FxDoc, uint32_t a_ImageId, Blob& a_Blob, const std::string& a_SourcePath)
{
	HeaderTexture ht;
    auto image = a_FxDoc.images[a_ImageId];

	if (!image.IsEmbeddedResource() && !image.uri.empty())
	{
		auto p = fx::gltf::detail::GetDocumentRootPath(a_SourcePath) + "/" + image.uri;
		std::ifstream ifs(p, std::ios::in | std::ios::binary | std::ios::ate);
		auto size = ifs.tellg();

		ifs.seekg(0, ifs.beg);

		a_Blob.ExtendToFit(size);
		ifs.read(reinterpret_cast<char*>(reinterpret_cast<uint64_t>(a_Blob.m_Data.data()) + a_Blob.m_Offset), size);
		ht.m_Offset = a_Blob.m_Offset;
		ht.m_Size = size;
		a_Blob.m_Offset += size;
	}
	else
	{
        if (image.IsEmbeddedResource())
        {
			std::vector<uint8_t> v;
			image.MaterializeData(v);

			ht.m_Offset = a_Blob.Write(v.data(), v.size());
			ht.m_Size = v.size();
        }
		else
		{
			fx::gltf::BufferView const& bufferView = a_FxDoc.bufferViews[image.bufferView];
			fx::gltf::Buffer const& buffer = a_FxDoc.buffers[bufferView.buffer];

			ht.m_Offset = a_Blob.Write(&buffer.data[bufferView.byteOffset], bufferView.byteLength);
			ht.m_Size = bufferView.byteLength;
		}
	}
	return ht;
}

LumenPTModelConverter::HeaderPrimitive LumenPTModelConverter::PrimitiveToBlob(const fx::gltf::Document& a_FxDoc,
    const fx::gltf::Primitive& a_FxPrimitive, Blob& a_Blob)
{
	// Extract the binary data of the primitive
	std::vector<uint8_t> posBinary, texBinary, norBinary, tanBinary;
	// For each attribute in the primitive
	for (auto& fxAttribute : a_FxPrimitive.attributes)
	{
		if (fxAttribute.first == "POSITION")
		{
			// store position accessor
			//fxAttribute.second

			posBinary = LoadBinary(a_FxDoc, fxAttribute.second);
			//make vertex buffer of this, make function

		}
		else if (fxAttribute.first == "TEXCOORD_0")
		{
			texBinary = LoadBinary(a_FxDoc, fxAttribute.second);
		}
		else if (fxAttribute.first == "COLOR_0")
		{

		}
		else if (fxAttribute.first == "NORMAL")
		{
			norBinary = LoadBinary(a_FxDoc, fxAttribute.second);
		}
		else if (fxAttribute.first == "TANGENT")
		{
			tanBinary = LoadBinary(a_FxDoc, fxAttribute.second);
		}

	}

	// Get binary data for the primitive's indices if those are included
	auto indexBufferAcc = a_FxDoc.accessors[a_FxPrimitive.indices];
	std::vector<unsigned char> indexBin = LoadBinary(a_FxDoc, a_FxPrimitive.indices);
	auto indexSize = GetComponentSize(indexBufferAcc); // indices are are always a single component
	assert(indexSize <= 4);

    if(tanBinary.empty())
    {
		tanBinary = GenerateTangentBinary(posBinary, norBinary, texBinary, indexBin, indexSize);
    }


	InterleaveInput interleave;
	interleave.m_Pos = &posBinary;
	interleave.m_Tex = &texBinary;
	interleave.m_Normal = &norBinary;
	interleave.m_Tangent = &tanBinary;

	auto interleaved = InterleaveVertexBuffers(interleave);

	HeaderPrimitive hp;

	hp.m_VertexBufferOffset = a_Blob.Write(interleaved.data(), interleaved.size());
	hp.m_VertexBufferSize = interleaved.size();

	hp.m_IndexBufferOffset = a_Blob.Write(indexBin.data(), indexBin.size());
	hp.m_IndexBufferSize = indexBin.size();
	hp.m_IndexSize = indexSize;

	hp.m_MaterialId = a_FxPrimitive.material;

	return hp;
}

std::vector<uint8_t> LumenPTModelConverter::GenerateTangentBinary(std::vector<uint8_t>& a_PosBinary, std::vector<uint8_t>& a_NormalBinary,
    std::vector<uint8_t>& a_TexBinary, std::vector<uint8_t>& a_IndexBinary, uint32_t a_IndexSize)
{
	VectorView<glm::vec3, uint8_t> posView(a_PosBinary);
	VectorView<glm::vec2, uint8_t> texView(a_TexBinary);
	VectorView<glm::vec3, uint8_t> normalView(a_NormalBinary);
	VectorView<uint32_t, uint8_t> indexView32(a_IndexBinary);
	VectorView<uint16_t, uint8_t> indexView16(a_IndexBinary);
	const auto numIndices = a_IndexBinary.size() / a_IndexSize;
	std::vector<uint8_t> tanBinary;
    tanBinary.resize(numIndices * sizeof(glm::vec4));

	VectorView<glm::vec4, uint8_t> tangentView(tanBinary);

	//////Invert the V coordinates
	//for (auto i = 0; i < texView.Size(); ++i)
	//{
	//	texView[i].y = 1.f - texView[i].y;
	//}

	//Default UV coordinates if none are specified.
	const glm::vec2 defaultUv[3]{ {1.f, 1.f}, {0.f, 1.f}, {1.f, 0.f} };

	//Loop over every triangle in the index buffer.
	for (auto index = 0u; index < numIndices; index += 3)
	{
		//Retrieve the indices from the index buffer.
		unsigned int indices[3];

		if (a_IndexSize == 2)
		{
			indices[0] = indexView16[index + 0];
			indices[1] = indexView16[index + 1];
			indices[2] = indexView16[index + 2];
		}
		else
		{
			indices[0] = indexView32[index + 0];
			indices[1] = indexView32[index + 1];
			indices[2] = indexView32[index + 2];
		}

		//Can't have a triangle that has two points the same. That's called a line.
		if (indices[0] == indices[1] || indices[1] == indices[2] || indices[2] == indices[0])
		{
			printf("Warning: Invalid vertex indices found.\n");
		}

		//Thanks to opengl-tutorial.com
		const glm::vec3* v0 = &posView[indices[0]];
		const glm::vec3* v1 = &posView[indices[1]];
		const glm::vec3* v2 = &posView[indices[2]];

		//UV coords per vertex.
		const glm::vec2* uv0 = nullptr;
		const glm::vec2* uv1 = nullptr;
		const glm::vec2* uv2 = nullptr;

		//No uv specified, use default.
		if (texView.Empty())
		{
			uv0 = &defaultUv[0];
			uv1 = &defaultUv[1];
			uv2 = &defaultUv[2];
		}
		//Specified, so look up.
		else
		{
			uv0 = &texView[indices[0]];
			uv1 = &texView[indices[1]];
			uv2 = &texView[indices[2]];
		}

		//Some meshes have invalid UVs defined. In those cases, use the defaults.
		float deltaUv0 = fabsf(glm::length(*uv0 - *uv1));
		float deltaUv1 = fabsf(glm::length(*uv0 - *uv2));
		float deltaUv2 = fabsf(glm::length(*uv2 - *uv1));
		constexpr float epsilon = std::numeric_limits<float>::epsilon();

		//Ensure that the UV coordinates have a volume.
		if (deltaUv0 < epsilon || deltaUv1 < epsilon || deltaUv2 < epsilon)
		{
			uv0 = &defaultUv[0];
			uv1 = &defaultUv[1];
			uv2 = &defaultUv[2];
		}

		glm::vec3 deltaPos1 = *v1 - *v0;
		glm::vec3 deltaPos2 = *v2 - *v0;
		glm::vec2 deltaUV1 = *uv1 - *uv0;
		glm::vec2 deltaUV2 = *uv2 - *uv0;

        ////If the cross is 0, it means the texture coords span an infinitely flat line. Set to default to avoid pain.
        const float cross = deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x;
        if (cross == 0)
        {
        	uv0 = &defaultUv[0];
        	uv1 = &defaultUv[1];
        	uv2 = &defaultUv[2];
        	deltaUV1 = *uv1 - *uv0;
        	deltaUV2 = *uv2 - *uv0;
        }

		//Generate the tangent vector.
        glm::vec3 t = (deltaUV2.t * deltaPos1 - deltaUV1.t * deltaPos2) / (deltaUV1.s * deltaUV2.t - deltaUV2.s * deltaUV1.t);

		for(int i = 0; i < 3; ++i)
		{
			//Take the tangent, and make it perpendicular to the normal.
			glm::vec3 tangent = t;
			glm::vec3 ng = normalize(normalView[indices[i]]);
			tangent = normalize(tangent - ng * dot(ng, tangent));

			//Store the tangent for each index.
			tangentView[indices[i]] = glm::vec4(tangent, 1.f);
		}


		//float cross = deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x;

		////If the cross is 0, it means the texture coords span an infinitely flat line. Set to default to avoid pain.
		//if (cross == 0)
		//{
		//	uv0 = &defaultUv[0];
		//	uv1 = &defaultUv[1];
		//	uv2 = &defaultUv[2];
		//	deltaPos1 = *v1 - *v0;
		//	deltaPos2 = *v2 - *v0;
		//	deltaUV1 = *uv1 - *uv0;
		//	deltaUV2 = *uv2 - *uv0;
		//	cross = deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x;
		//}

		////If cross is 0, it means the texture coordinates are a single point for two corners.
		//assert(cross != 0);

		//float r = 1.0f / cross;

		//glm::vec3 tangent = glm::vec3((deltaPos1 * deltaUV2.y - deltaPos2 * deltaUV1.y) * r);

		//assert(!isnan(tangent.x));
		//assert(!isnan(tangent.y));
		//assert(!isnan(tangent.z));
		//assert(glm::length(glm::vec3(tangent.x, tangent.y, tangent.z)) > 0.f);
		////
		//if (isnan(tangent.x) || isnan(tangent.y) || isnan(tangent.z) || isinf(tangent.x) || isinf(tangent.y) || isinf(tangent.z))
		//{
		//	tangent = glm::vec3(1.f, 0.f, 0.f);
		//}

		////Normalize the tangent.
		//tangent = glm::normalize(tangent);

		////Ensure the tangent is valid.
		//assert(glm::length(tangent) > 0.f);

		//auto tanVec4 = glm::vec4(tangent, 1.f);

		////Put in the output buffer. Same tangent for every vertex that was processed.
		//tangentView[index1] = tanVec4;
		//tangentView[index2] = tanVec4;
		//tangentView[index3] = tanVec4;
	}
	

	return tanBinary;
}

std::vector<char> LumenPTModelConverter::InterleaveVertexBuffers(InterleaveInput& a_Input)
{
	VectorView<glm::vec3, uint8_t> posView(*a_Input.m_Pos);
	VectorView<glm::vec2, uint8_t> texView(*a_Input.m_Tex);
	VectorView<glm::vec3, uint8_t> normalView(*a_Input.m_Normal);
	VectorView<glm::vec4, uint8_t> tangentView(*a_Input.m_Tangent);

	std::vector<char> interleavedBinary(sizeof(Vertex) * posView.Size());
	VectorView<Vertex, char> interleavedView(interleavedBinary);

	for (size_t i = 0; i < posView.Size(); i++)
	{
		interleavedView[i].m_Position = make_float3(posView[i].x, posView[i].y, posView[i].z);
        if (!a_Input.m_Tex->empty())
		    interleavedView[i].m_UVCoord = make_float2(texView[i].x, texView[i].y);
		if (!a_Input.m_Normal->empty())
			interleavedView[i].m_Normal = make_float3(normalView[i].x, normalView[i].y, normalView[i].z);
		if (!a_Input.m_Tangent->empty())
			interleavedView[i].m_Tangent = make_float4(tangentView[i].x, tangentView[i].y, tangentView[i].z, tangentView[i].w);
	}

	std::vector<Vertex> arr(interleavedView.Size());

	memcpy(arr.data(), interleavedBinary.data(), interleavedBinary.size());

	return interleavedBinary;
}

LumenPTModelConverter::HeaderScene LumenPTModelConverter::MakeScene(const fx::gltf::Document& a_FxDoc, const fx::gltf::Scene& a_Scene)
{
	HeaderScene hscene;

	
	int index = 0;
    for (auto rootNode : a_Scene.nodes)
    {
		printf("[File Conversion] Now loading root node %i of %i.\n", index, static_cast<int>(a_Scene.nodes.size()) - 1);
    	
		LoadNode(a_FxDoc, rootNode, hscene);

		printf("[File Conversion] Done loading root node %i of %i.\n", index, static_cast<int>(a_Scene.nodes.size()) - 1);
		++index;
    }

	hscene.m_Name = a_Scene.name;
	hscene.m_Header.m_NumMeshes = hscene.m_Meshes.size();
	hscene.m_Header.m_NameLength = hscene.m_Name.size();

	return hscene;
}

void LumenPTModelConverter::LoadNode(const fx::gltf::Document& a_FxDoc, uint32_t a_NodeId, HeaderScene& a_Scene,
    glm::mat4 a_ParentTransform)
{
	const auto& node = a_FxDoc.nodes[a_NodeId];

	auto localTransform = LoadNodeTransform(node);
	auto worldTransform = a_ParentTransform * localTransform;

	auto mat = worldTransform.GetTransformationMatrix();

    if (node.mesh != -1)
    {
		auto& m = a_Scene.m_Meshes.emplace_back();
		m.m_Header.m_MeshId = node.mesh;

		memcpy(m.m_Header.m_Transform, &mat[0], sizeof(glm::mat4));
		m.m_Name = node.name;
		m.m_Header.m_NameLength = node.name.size();
    }

	int index = 0;

    for (auto& ch : node.children)
    {
		printf("[File Conversion] Now loading child node %i of %i.\n", index, static_cast<int>(node.children.size()) - 1);
		LoadNode(a_FxDoc, ch, a_Scene, worldTransform);
		++index;
    }
}

Lumen::Transform LumenPTModelConverter::LoadNodeTransform(const fx::gltf::Node& a_Node)
{
    Lumen::Transform transform = glm::make_mat4(&a_Node.matrix[0]);

    if (transform.GetTransformationMatrix() == glm::identity<glm::mat4>())
    {
		auto translation = glm::vec3(
			a_Node.translation[0],
			a_Node.translation[1],
			a_Node.translation[2]
		);

		auto rotation = glm::quat(
			a_Node.rotation[3],
			a_Node.rotation[0],
			a_Node.rotation[1],
			a_Node.rotation[2]
		);

		auto scale = glm::vec3(
			a_Node.scale[0],
			a_Node.scale[1],
			a_Node.scale[2]
		);
		transform.SetPosition(translation);
		transform.SetRotation(rotation);
		transform.SetScale(scale);
    }

	return transform;
}


std::vector<uint8_t> LumenPTModelConverter::LoadBinary(const fx::gltf::Document& a_Doc, uint32_t a_AccessorIndx)
{
	std::vector<unsigned char> data;
	// Load raw data at accessor index
	auto bufferAccessor = a_Doc.accessors[a_AccessorIndx];
	auto bufferView = a_Doc.bufferViews[bufferAccessor.bufferView];
	auto buffer = a_Doc.buffers[bufferView.buffer];

	uint32_t compCount = GetComponentCount(bufferAccessor);
	uint32_t compSize = GetComponentSize(bufferAccessor);
	uint32_t attSize = compCount * compSize;

	// Sometimes bufferView.byteStride is 0, so we pick the biggest of the two
	// That way this works for interleaved buffers as well
	uint32_t stride = std::max(attSize, bufferView.byteStride);

	// Resize the output buffer to fit all the data that will be extracted
	data.resize(bufferAccessor.count * static_cast<uint64_t>(attSize));

	// Offset from which to read in the buffer
	auto bufferOffset = 0;
	// Buffer offset specified in the file
	auto gltfBufferOffset = bufferView.byteOffset + bufferAccessor.byteOffset;

	// Copy over the binary data via a loop
	for (uint32_t i = 0; i < bufferAccessor.count; i++)
	{
		memcpy(data.data() + bufferOffset, buffer.data.data() + gltfBufferOffset + bufferOffset, attSize);
		bufferOffset += stride;
	}

	return data;
}


uint32_t LumenPTModelConverter::GetComponentSize(fx::gltf::Accessor& a_Accessor)
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
		return 0;
	}

}

uint32_t LumenPTModelConverter::GetComponentCount(fx::gltf::Accessor& a_Accessor)
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
	case fx::gltf::Accessor::Type::Vec4:
		return 4;
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
		assert(0);
		return 0;
	}
}