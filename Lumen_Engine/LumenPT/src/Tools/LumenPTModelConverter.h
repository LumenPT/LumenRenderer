#pragma once

#include "gltf.h"

//#include "../vendor/fx/gltf.h"

#include "Lumen/ModelLoading/SceneManager.h"

class LumenPTModelConverter
{
public:

    LumenPTModelConverter() {};

    ~LumenPTModelConverter() {};

    Lumen::SceneManager::GLTFResource ConvertGLTF(std::string a_SourcePath);
    
    Lumen::SceneManager::GLTFResource LoadFile(std::string a_SourcePath);

    void SetRendererRef(LumenRenderer& a_Renderer);

    const static inline std::string ms_ExtensionName = ".ollad";
    
private:
    struct Blob 
    {
        Blob()
            : m_Data(1024)
            , m_Offset(0) {}
        void ExtendToFit(uint64_t a_ChunkSize) 
        {            
            auto s = m_Data.size();
            while (s < m_Offset + a_ChunkSize)
            {
                s *= 2;
            }

            m_Data.resize(s);
        }

        uint64_t Write(const void* a_Src, uint64_t a_Size) 
        {
            ExtendToFit(a_Size);
            memcpy(m_Data.data() + m_Offset, a_Src, a_Size);

            auto offset = m_Offset;
            m_Offset += a_Size;
            return offset;
        }

        void Trim()
        {
            m_Data.erase(m_Data.begin() + m_Offset, m_Data.end());
        }

        std::vector<char> m_Data;
        union
        {
            uint64_t m_Offset;
            uint64_t m_Size;
        };
    };

    enum TextureType : uint64_t
    {
        EUnspecified = 0,
        EDiffuse = 1,
        ENormal,
        EEmissive,
        EMetalRoughness
    };

    struct HeaderTexture
    {
        uint64_t m_Offset;
        uint64_t m_Size;
        uint64_t m_TextureType;
    };
    struct HeaderMaterial
    {
        HeaderMaterial()
            : m_DiffuseTextureId(-1)
            , m_NormalMapId(-1)
            , m_EmissiveTextureId(-1)
            , m_MetallicRoughnessTextureId(-1)
        {}
        float m_Color[4];
        float m_Emission[3];
        int32_t m_DiffuseTextureId;
        int32_t m_NormalMapId;
        int32_t m_MetallicRoughnessTextureId;
        int32_t m_EmissiveTextureId;

        int32_t m_TransmittanceTextureId;
        int32_t m_Transmittance
    };

    struct HeaderPrimitive 
    {
        uint64_t m_VertexBufferOffset;
        uint64_t m_VertexBufferSize;
        uint64_t m_IndexBufferOffset;
        uint64_t m_IndexBufferSize;
        uint32_t m_IndexSize;
        uint32_t m_MaterialId;
    };

    struct HeaderMesh
    {
        struct
        {
            uint32_t m_NumPrimitives;
        } m_Header;
        std::vector<HeaderPrimitive> m_Primitives;
    };

    struct HeaderMeshInstance
    {
        int32_t m_MeshId;
        float m_Transform[16];
    };

    struct HeaderScene
    {
        struct
        {
            uint32_t m_NumMeshes;
        } m_Header;
        std::vector<HeaderMeshInstance> m_Meshes;
    };

    struct FileContent
    {
        std::vector<HeaderTexture> m_Textures;
        std::vector<HeaderMaterial> m_Materials;
        std::vector<HeaderMesh> m_Meshes;
        std::vector<HeaderScene> m_Scenes;

        Blob m_Blob;
    };

    struct Header
    {
        uint64_t m_Size;
        Blob m_Binary;
    };

    static FileContent GenerateContent(const fx::gltf::Document& a_FxDoc, const std::string& a_SourcePath);
    static Header GenerateHeader(const FileContent& a_Content);
    static void OutputToFile(const Header& a_Header, const Blob& a_Binary, const std::string& a_DestPath);

    static HeaderTexture TextureToBlob(const fx::gltf::Document& a_FxDoc, uint32_t a_ImageId, Blob& a_Blob, const std::string& a_SourcePath);

    static HeaderPrimitive PrimitiveToBlob(const fx::gltf::Document& a_FxDoc, const fx::gltf::Primitive& a_FxPrimitive, Blob& a_Blob);

    static std::vector<uint8_t> GenerateTangentBinary(std::vector<uint8_t>& a_PosBinary, std::vector<uint8_t>& a_TexBinary, std::vector<uint8_t>& a_IndexBinary, uint32_t a_IndexSize);

    struct InterleaveInput
    {
        std::vector<uint8_t>* m_Pos;
        std::vector<uint8_t>* m_Tex;
        std::vector<uint8_t>* m_Normal;
        std::vector<uint8_t>* m_Tangent;
    };

    static std::vector<char> InterleaveVertexBuffers(InterleaveInput& a_Input);

    static HeaderScene MakeScene(const fx::gltf::Document& a_FxDoc, const fx::gltf::Scene& a_Scene);
    static void LoadNode(const fx::gltf::Document& a_FxDoc, uint32_t a_NodeId, HeaderScene& a_Scene, glm::mat4 a_ParentTransform = glm::identity<glm::mat4>());
    static Lumen::Transform LoadNodeTransform(const fx::gltf::Node& a_Node);


    static std::vector<uint8_t> LoadBinary(const fx::gltf::Document& a_Doc, uint32_t a_AccessorIndx);
    static uint32_t GetComponentCount(fx::gltf::Accessor& a_Accessor); // Return how many components the accessor uses
    static uint32_t GetComponentSize(fx::gltf::Accessor& a_Accessor); // Return the size of the components used by the accessor

    LumenRenderer* m_RendererRef;

    std::shared_ptr<Lumen::ILumenTexture> m_DefaultDiffuseTexture;
    std::shared_ptr<Lumen::ILumenTexture> m_DefaultMetalRoughnessTexture;
    std::shared_ptr<Lumen::ILumenTexture> m_DefaultNormalTexture;
    std::shared_ptr<Lumen::ILumenTexture> m_DefaultEmissiveTexture;

};

