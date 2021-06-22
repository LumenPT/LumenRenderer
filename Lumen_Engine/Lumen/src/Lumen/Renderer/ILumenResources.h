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
        virtual void SetEmission(const glm::vec3& a_EmssivionVal = glm::vec3(0.0f, 0.0f, 0.0f)) = 0;
        virtual void SetEmissiveTexture(std::shared_ptr<ILumenTexture> a_EmissiveTexture) = 0;
        virtual void SetMetalRoughnessTexture(std::shared_ptr<ILumenTexture> a_MetalRoughnessTexture) = 0;
        virtual void SetNormalTexture(std::shared_ptr<ILumenTexture> a_NormalTexture) = 0;

        //Disney BSDF stuff
        virtual void SetClearCoatTexture(std::shared_ptr<ILumenTexture> a_Texture) = 0;
        virtual void SetClearCoatRoughnessTexture(std::shared_ptr<ILumenTexture> a_Texture) = 0;
        virtual void SetClearCoatFactor(float a_Factor) = 0;
        virtual void SetClearCoatRoughnessFactor(float a_Factor) = 0;

        virtual void SetLuminance(float a_Factor) = 0;
        virtual void SetSheenFactor(float a_Factor) = 0;
        virtual void SetSheenTintFactor(float a_Factor) = 0;

        virtual void SetAnisotropic(float a_Factor) = 0;

        virtual void SetTintTexture(std::shared_ptr<ILumenTexture> a_Texture) = 0;
        virtual void SetTintFactor(const glm::vec3& a_Factor) = 0;

        virtual void SetTransmissionTexture(std::shared_ptr<ILumenTexture> a_Texture) = 0;
        virtual void SetTransmissionFactor(float a_Factor) = 0;
        virtual void SetTransmittanceFactor(const glm::vec3& a_Factor) = 0;
        virtual void SetIndexOfRefraction(float a_Factor) = 0;

        virtual void SetSpecularFactor(float a_Factor) = 0;
        virtual void SetSpecularTintFactor(float a_Factor) = 0;
        virtual void SetSubSurfaceFactor(float a_Factor) = 0;

        virtual void SetMetallicFactor(float a_Factor) = 0;
        virtual void SetRoughnessFactor(float a_Factor) = 0;

        // ---------------------------------------------------------------------------------------------

        virtual float GetClearCoatFactor() = 0;
        virtual float GetClearCoatRoughnessFactor() = 0;
        
        virtual float GetLuminance() = 0;
        virtual float GetSheenFactor() = 0;
        virtual float GetSheenTintFactor() = 0;
       
        virtual float GetAnisotropic() = 0;
       
        virtual glm::vec3 GetTintFactor() = 0;
       
        virtual float GetTransmissionFactor() = 0;
        virtual glm::vec3 GetTransmittanceFactor() = 0;
        virtual float GetIndexOfRefraction() = 0;
       
        virtual float GetSpecularFactor() = 0;
        virtual float GetSpecularTintFactor() = 0;
        virtual float GetSubSurfaceFactor() = 0;
       
        virtual float GetMetallicFactor() = 0;
        virtual float GetRoughnessFactor() = 0;

        //TODO add disney BSDF getter functions.

        virtual glm::vec4 GetDiffuseColor() const = 0;
        virtual glm::vec3 GetEmissiveColor() const = 0;
        virtual ILumenTexture& GetDiffuseTexture() const = 0;
        virtual ILumenTexture& GetEmissiveTexture() const = 0;
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
        ILumenMesh(std::vector<std::shared_ptr<ILumenPrimitive>>& a_Primitives)
            : m_Primitives(std::move(a_Primitives)) 
        {
            for (auto& prim : m_Primitives)
            {
                if (prim->m_ContainEmissive)
                {
                    m_Emissive = true;
                    break;
                }
            }
        };

        std::vector<std::shared_ptr<ILumenPrimitive>> m_Primitives;
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
