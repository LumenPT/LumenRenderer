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

namespace fs = std::filesystem;
using namespace fx::gltf;

Lumen::SceneManager::GLTFResource LumenPTModelConverter::ConvertGLTF(std::string a_SourcePath)
{
	auto workDir = fs::current_path();
	auto p = workDir.string().append(a_SourcePath);

	fs::path sp(p);

	fx::gltf::Document fxDoc;

	if (sp.extension() == ".gltf")
		fxDoc = LoadFromText(p);
	else if (sp.extension() == ".glb")
		fxDoc = LoadFromBinary(p);
	


	auto content = GenerateContent(fxDoc, p);
	auto header = GenerateHeader(content);

	p.erase(p.begin() + p.find('.'), p.end());
	std::string destPath = p.append(ms_ExtensionName);

	OutputToFile(header, content.m_Blob, destPath);

	return LoadFile(destPath);
}

Lumen::SceneManager::GLTFResource LumenPTModelConverter::LoadFile(std::string a_SourcePath)
{
	Lumen::SceneManager::GLTFResource res;

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

		printf("shit's loaded like my nuts :)....... :(");

		// Move the cursor back to the beginning of the header, right after the number signifying the size of the header
		ifs.seekg(sizeof(headerSize), ifs.beg);

		uint64_t numTex;
		ifs.read(reinterpret_cast<char*>(&numTex), sizeof(uint64_t));

		for (size_t i = 0; i < numTex; i++)
		{
			HeaderTexture ht;
			ifs.read(reinterpret_cast<char*>(&ht), sizeof(ht));

			int32_t x, y, c;

			// TODO: This seems to be the actual bottleneck in the loading process.
			// Could maybe move the image decompression to the conversion process.
			auto texData = stbi_load_from_memory(reinterpret_cast<uint8_t*>(&dataBin[ht.m_Offset]), ht.m_Size, &x, &y, &c, 4);

			// TODO: Create a texture object from this data
		}

		uint64_t numMat;
		ifs.read(reinterpret_cast<char*>(&numMat), sizeof(uint64_t));

		for (size_t i = 0; i < numMat; i++)
		{
			HeaderMaterial hm;
			ifs.read(reinterpret_cast<char*>(&hm), sizeof(hm));

			// TODO: Create a material object from this data	
		}

		uint64_t numMeshes;
		ifs.read(reinterpret_cast<char*>(&numMeshes), sizeof(uint64_t));

		for (size_t i = 0; i < numMeshes; i++)
		{
			decltype(HeaderMesh::m_Header) meshHeader;
			ifs.read(reinterpret_cast<char*>(&meshHeader), sizeof(meshHeader));

			for (size_t j = 0; j < meshHeader.m_NumPrimitives; j++)
			{
				HeaderPrimitive primitive;

				ifs.read(reinterpret_cast<char*>(&primitive), sizeof(primitive));

				printf("test");
				// TODO: Create a primitive from this data
			}
		}
	}
	else
		res.m_Path = ""; // Empty path means no file 

	return res;
}

LumenPTModelConverter::FileContent LumenPTModelConverter::GenerateContent(const fx::gltf::Document& a_FxDoc, const std::string& a_SourcePath)
{
	FileContent fc;

    for (uint32_t i = 0; i < a_FxDoc.images.size(); i++)
    {
		fc.m_Textures.push_back(TextureToBlob(a_FxDoc, i, fc.m_Blob, a_SourcePath));
    }

    for (auto& material : a_FxDoc.materials)
    {
		auto& m = fc.m_Materials.emplace_back();

		m.m_Color[0] = material.pbrMetallicRoughness.baseColorFactor[0];
		m.m_Color[1] = material.pbrMetallicRoughness.baseColorFactor[1];
		m.m_Color[2] = material.pbrMetallicRoughness.baseColorFactor[2];
		m.m_Color[3] = material.pbrMetallicRoughness.baseColorFactor[3];

		m.m_DiffuseTextureId = material.pbrMetallicRoughness.baseColorTexture.index;
		m.m_NormalMapId = material.normalTexture.index;
    }

    for (auto& mesh : a_FxDoc.meshes)
    {
		auto& m = fc.m_Meshes.emplace_back();
        for (auto& primitive : mesh.primitives)
        {
			m.m_Primitives.push_back(PrimitiveToBlob(a_FxDoc, primitive, fc.m_Blob));
        }
		m.m_Header.m_NumPrimitives = m.m_Primitives.size();
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


	//If no tangents were loaded, generate them.
	if (tanBinary.empty())
	{
		tanBinary = GenerateTangentBinary(posBinary, texBinary, indexBin, indexSize);
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

std::vector<uint8_t> LumenPTModelConverter::GenerateTangentBinary(std::vector<uint8_t>& a_PosBinary,
    std::vector<uint8_t>& a_TexBinary, std::vector<uint8_t>& a_IndexBinary, uint32_t a_IndexSize)
{
	VectorView<glm::vec3, uint8_t> vertexView(a_PosBinary);
	VectorView<glm::vec2, uint8_t> texView(a_TexBinary);
	VectorView<uint32_t, uint8_t> indexView32(a_IndexBinary);
	VectorView<uint16_t, uint8_t> indexView16(a_IndexBinary);
	const auto numIndices = a_IndexBinary.size() / a_IndexSize;
	std::vector<uint8_t> tanBinary;
    tanBinary.reserve(numIndices);
    tanBinary.resize(numIndices * sizeof(glm::vec4));

	VectorView<glm::vec4, uint8_t> tangentView(tanBinary);

	////Invert the V coordinates
	for (auto i = 0; i < texView.Size(); ++i)
	{
		texView[i].y = 1.f - texView[i].y;
	}

	//Loop over every triangle in the index buffer.
	for (auto index = 0u; index < numIndices; index += 3)
	{
		//Retrieve the indices from the index buffer.
		unsigned int index1;
		unsigned int index2;
		unsigned int index3;

		if (a_IndexSize == 2)
		{
			index1 = indexView16[index + 0];
			index2 = indexView16[index + 1];
			index3 = indexView16[index + 2];
		}
		else
		{
			index1 = indexView32[index + 0];
			index2 = indexView32[index + 1];
			index3 = indexView32[index + 2];
		}

		//Can't have a triangle that has two points the same. That's called a line.
		if (index1 == index2 || index2 == index3 || index3 == index1)
		{
			printf("Warning: Invalid vertex indices found.\n");
		}

		//Thanks to opengl-tutorial.com
		const glm::vec3* v0 = &vertexView[index1];
		const glm::vec3* v1 = &vertexView[index2];
		const glm::vec3* v2 = &vertexView[index3];
		glm::vec2* uv0 = &texView[index1];
		glm::vec2* uv1 = &texView[index2];
		glm::vec2* uv2 = &texView[index3];

		glm::vec3 deltaPos1 = *v1 - *v0;
		glm::vec3 deltaPos2 = *v2 - *v0;

		glm::vec2 deltaUV1 = *uv1 - *uv0;
		glm::vec2 deltaUV2 = *uv2 - *uv0;

		float cross = deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x;

		//If cross is 0, it means the texture coordinates are a single point for two corners.
		assert(cross != 0);

		float r = 1.0f / cross;

		glm::vec3 tangent = glm::vec3((deltaPos1 * deltaUV2.y - deltaPos2 * deltaUV1.y) * r);

		assert(!isnan(tangent.x));
		assert(!isnan(tangent.y));
		assert(!isnan(tangent.z));
		assert(glm::length(glm::vec3(tangent.x, tangent.y, tangent.z)) > 0.f);
		//
		if (isnan(tangent.x) || isnan(tangent.y) || isnan(tangent.z) || isinf(tangent.x) || isinf(tangent.y) || isinf(tangent.z))
		{
			tangent = glm::vec3(1.f, 0.f, 0.f);
		}

		//Normalize the tangent.
		tangent = glm::normalize(tangent);

		auto tanVec4 = glm::vec4(tangent, 1.f);

		//Put in the output buffer. Same tangent for every vertex that was processed.
		tangentView[index1] = tanVec4;
		tangentView[index2] = tanVec4;
		tangentView[index3] = tanVec4;
	}

	return tanBinary;
}

std::vector<char> LumenPTModelConverter::InterleaveVertexBuffers(InterleaveInput& a_Input)
{
	VectorView<float3, uint8_t> posView(*a_Input.m_Pos);
	VectorView<float2, uint8_t> texView(*a_Input.m_Tex);
	VectorView<float3, uint8_t> normalView(*a_Input.m_Normal);
	VectorView<float4, uint8_t> tangentView(*a_Input.m_Tangent);

	std::vector<char> interleavedBinary(sizeof(Vertex) * posView.Size());
	VectorView<Vertex, char> interleavedView(interleavedBinary);

	for (size_t i = 0; i < posView.Size(); i++)
	{
		interleavedView[i].m_Position = posView[i];
		interleavedView[i].m_UVCoord = texView[i];
		interleavedView[i].m_Normal = normalView[i];
		interleavedView[i].m_Tangent = tangentView[i];
	}

	return interleavedBinary;
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
	data.resize(bufferView.byteLength / stride * static_cast<uint64_t>(attSize));

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