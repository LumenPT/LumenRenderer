#pragma once
#include <glm/fwd.hpp>
#include <memory>
#include <vector>
#include "glm/vec3.hpp"
#include "glm/vec4.hpp"


namespace Lumen
{
    // Interface class for textures
	class ILumenTexture
	{
	public:
		virtual ~ILumenTexture() = default;
	};

    // Interface class for materials
    class ILumenMaterial
    {
    public:
        virtual ~ILumenMaterial() = default;
        virtual void SetDiffuseColor(const glm::vec4& a_NewDiffuseColor) = 0;
        virtual void SetDiffuseTexture(std::shared_ptr<ILumenTexture> a_NewDiffuseTexture) = 0;
        virtual void SetEmission(const glm::vec3& a_EmssivionVal = glm::vec3(1.0f, 1.0f, 1.0f)) = 0;
        virtual void SetEmissiveTexture(std::shared_ptr<ILumenTexture> a_EmissiveTexture) = 0;
        virtual void SetMetalRoughnessTexture(std::shared_ptr<ILumenTexture> a_MetalRoughnessTexture) = 0;
        virtual void SetNormalTexture(std::shared_ptr<ILumenTexture> a_NormalTexture) = 0;
    	
        virtual glm::vec4 GetDiffuseColor() const = 0;
        virtual ILumenTexture& GetDiffuseTexture() const = 0;
    };

    // Interface class for primitives
    class ILumenPrimitive
    {
    public:
        virtual ~ILumenPrimitive() {};
        std::shared_ptr<ILumenMaterial> m_Material;
        bool m_ContainEmissive;
        unsigned int m_NumLights;   //number of triangles covered by emissive mat
    };

    // Interface class for meshes
    class ILumenMesh
    {
    public:
        ILumenMesh(std::vector<std::unique_ptr<ILumenPrimitive>>& a_Primitives)
            : m_Primitives(std::move(a_Primitives)) 
        {
            for (auto& prim : a_Primitives)
            {
                if (prim->m_ContainEmissive)
                {
                    m_Emissive = true;
                    break;
                }
            }
        };

        std::vector<std::unique_ptr<ILumenPrimitive>> m_Primitives;
        const bool& GetEmissiveness() { return m_Emissive; };
    private:
        bool m_Emissive;
    };

    // Interface class for volumes
    class ILumenVolume
    {
    public:
        virtual ~ILumenVolume() = default;
    };

}
