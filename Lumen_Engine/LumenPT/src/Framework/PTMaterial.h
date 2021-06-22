#pragma once

#include "Renderer/ILumenResources.h"

#include "glm/vec4.hpp"
#include "glm/vec3.hpp"

#include "Cuda/vector_types.h"

#include <memory>

#include "../Shaders/CppCommon/MaterialStructs.h"

struct DeviceMaterial;

// Pathtracer-specific implementation of the material class.
// Used to keep the materials for the path tracer up to date and correct.
class PTMaterial : public Lumen::ILumenMaterial
{
public:
    PTMaterial();

    // Returns a GPU pointer to a representation of the material.
    DeviceMaterial* GetDeviceMaterial() const;

    void SetDiffuseColor(const glm::vec4& a_NewDiffuseColor) override;
    void SetDiffuseTexture(std::shared_ptr<Lumen::ILumenTexture> a_NewDiffuseTexture) override;
    void SetEmission(const glm::vec3& a_EmissiveVal = glm::vec3( 0.0f, 0.0f, 0.0f)) override;
    void SetEmissiveTexture(std::shared_ptr<Lumen::ILumenTexture> a_EmissiveTexture) override;
    void SetMetalRoughnessTexture(std::shared_ptr<Lumen::ILumenTexture> a_MetalRoughnessTexture) override;
    void SetNormalTexture(std::shared_ptr<Lumen::ILumenTexture> a_NormalTexture) override;

    glm::vec4 GetDiffuseColor() const override;
    glm::vec3 GetEmissiveColor() const override;
    Lumen::ILumenTexture& GetDiffuseTexture() const override;
    Lumen::ILumenTexture& GetEmissiveTexture() const override;

private:

    // Conversion function from the CPU representation to the GPU representation.
    // Make sure to update this if you are expanding the materials beyond what already exists.
    DeviceMaterial CreateDeviceMaterial() const;

//Don't let the mouse see this or we're boned!
public:
    void SetClearCoatTexture(std::shared_ptr<Lumen::ILumenTexture> a_Texture) override;
    void SetClearCoatRoughnessTexture(std::shared_ptr<Lumen::ILumenTexture> a_Texture) override;
    void SetTransmissionTexture(std::shared_ptr<Lumen::ILumenTexture> a_Texture) override;
    void SetClearCoatFactor(float a_Factor) override;
    void SetClearCoatRoughnessFactor(float a_Factor) override;
    void SetIndexOfRefraction(float a_Factor) override;
    void SetTransmissionFactor(float a_Factor) override;
    void SetSpecularFactor(float a_Factor) override;
    void SetSpecularTintFactor(float a_Factor) override;
    void SetSubSurfaceFactor(float a_Factor) override;
    void SetLuminance(float a_Factor) override;
    void SetTintTexture(std::shared_ptr<Lumen::ILumenTexture> a_Texture) override;
    void SetTintFactor(const glm::vec3& a_Factor) override;
    void SetAnisotropic(float a_Factor) override;
    void SetSheenFactor(float a_Factor) override;
    void SetSheenTintFactor(float a_Factor) override;
    void SetTransmittanceFactor(const glm::vec3& a_Factor) override;
    void SetMetallicFactor(float a_Factor) override;
    void SetRoughnessFactor(float a_Factor) override;

    float GetClearCoatFactor() override { return m_MaterialData.GetClearCoat(); }
    float GetClearCoatRoughnessFactor() override { return 1.f - m_MaterialData.GetClearCoatGloss(); }

    float GetLuminance() override { return m_MaterialData.GetLuminance(); }
    float GetSheenFactor() override { return m_MaterialData.GetSheen(); }
    float GetSheenTintFactor() override { return m_MaterialData.GetSheenTint(); }

    float GetAnisotropic() override { return m_MaterialData.GetAnisotropic(); }

    glm::vec3 GetTintFactor() override
    {
        const float3 data = m_MaterialData.GetTint();
	    return glm::vec3(data.x, data.y, data.z);
    }

    float GetTransmissionFactor() override { return m_MaterialData.GetTransmission(); }
    glm::vec3 GetTransmittanceFactor() override
    {
        const auto data = m_MaterialData.GetTransmittance();
	    return glm::vec3(data.x, data.y, data.z);
    }
    float GetIndexOfRefraction() override { return m_MaterialData.GetRefractiveIndex(); }

    float GetSpecularFactor() override { return m_MaterialData.GetSpecular(); }
    float GetSpecularTintFactor() override { return m_MaterialData.GetSpecTint(); }
    float GetSubSurfaceFactor() override { return m_MaterialData.GetSubSurface(); }

    float GetMetallicFactor() override { return m_MaterialData.GetMetallic(); }
    float GetRoughnessFactor() override { return m_MaterialData.GetRoughness(); }

private:
    // Material data is kept here instead of the base class to account for API-specific implementation details   
    std::shared_ptr<class PTTexture> m_DiffuseTexture;
    std::shared_ptr<class PTTexture> m_EmissiveTexture;
    std::shared_ptr<class PTTexture> m_MetalRoughnessTexture;
    std::shared_ptr<class PTTexture> m_NormalTexture;

    std::shared_ptr<class PTTexture> m_TransmissionTexture;
    std::shared_ptr<class PTTexture> m_ClearCoatTexture;
    std::shared_ptr<class PTTexture> m_ClearCoatRoughnessTexture;
    std::shared_ptr<class PTTexture> m_TintTexture;

	//Tightly packed material data.
    MaterialData m_MaterialData;

    // A flag to keep track if the GPU representation of the material needs to be updated after something was changed
    mutable bool m_DeviceMaterialDirty;
    // A small GPU memory buffer to store the GPU representation in
    mutable std::unique_ptr<class MemoryBuffer> m_DeviceMemoryBuffer;
};