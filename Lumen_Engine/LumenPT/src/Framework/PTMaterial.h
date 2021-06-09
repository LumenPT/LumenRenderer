#pragma once

#include "Renderer/ILumenResources.h"

#include "glm/vec4.hpp"
#include "glm/vec3.hpp"

#include "Cuda/vector_types.h"

#include <memory>

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

    float GetClearCoatFactor() override { return m_ClearCoatFactor; }
    float GetClearCoatRoughnessFactor() override { return m_ClearCoatRoughnessFactor; }

    float GetLuminance() override { return m_Luminance; }
    float GetSheenFactor() override { return m_SheenFactor; }
    float GetSheenTintFactor() override { return m_SheenTintFactor; }

    float GetAnisotropic() override { return m_Anisotropic; }

    glm::vec3 GetTintFactor() override { return glm::vec3(m_TintFactor.x, m_TintFactor.y, m_TintFactor.z); }

    float GetTransmissionFactor() override { return m_TransmissionFactor; }
    glm::vec3 GetTransmittanceFactor() override { return glm::vec3(m_Transmittance.x, m_Transmittance.y, m_Transmittance.z); }
    float GetIndexOfRefraction() override { return m_IndexOfRefraction; }

    float GetSpecularFactor() override { return m_SpecularFactor; }
    float GetSpecularTintFactor() override { return m_SpecularTintFactor; }
    float GetSubSurfaceFactor() override { return m_SubSurfaceFactor; }

    float GetMetallicFactor() override { return m_MetallicFactor; }
    float GetRoughnessFactor() override { return m_RoughnessFactor; }

private:
    // Material data is kept here instead of the base class to account for API-specific implementation details   
    float4 m_DiffuseColor;
    float4 m_EmissiveColor;
    std::shared_ptr<class PTTexture> m_DiffuseTexture;
    std::shared_ptr<class PTTexture> m_EmissiveTexture;
    std::shared_ptr<class PTTexture> m_MetalRoughnessTexture;
    std::shared_ptr<class PTTexture> m_NormalTexture;

    std::shared_ptr<class PTTexture> m_TransmissionTexture;
    std::shared_ptr<class PTTexture> m_ClearCoatTexture;
    std::shared_ptr<class PTTexture> m_ClearCoatRoughnessTexture;
    std::shared_ptr<class PTTexture> m_TintTexture;

    float m_TransmissionFactor;
    float m_ClearCoatFactor;
    float m_ClearCoatRoughnessFactor;
    float m_IndexOfRefraction;
    float m_SpecularFactor;
    float m_SpecularTintFactor;
    float m_SubSurfaceFactor;
    float m_Luminance;
    float m_Anisotropic;
    float m_SheenFactor;
    float m_SheenTintFactor;
    float m_MetallicFactor;
    float m_RoughnessFactor;
    float3 m_TintFactor;
    float3 m_Transmittance;

    // A flag to keep track if the GPU representation of the material needs to be updated after something was changed
    mutable bool m_DeviceMaterialDirty;
    // A small GPU memory buffer to store the GPU representation in
    mutable std::unique_ptr<class MemoryBuffer> m_DeviceMemoryBuffer;
};