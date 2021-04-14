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
    Lumen::ILumenTexture& GetDiffuseTexture() const override;

private:

    // Conversion function from the CPU representation to the GPU representation.
    // Make sure to update this if you are expanding the materials beyond what already exists.
    DeviceMaterial CreateDeviceMaterial() const;

private:
    // Material data is kept here instead of the base class to account for API-specific implementation details   
    float4 m_DiffuseColor;
    float3 m_EmissiveColor;
    std::shared_ptr<class PTTexture> m_DiffuseTexture;
    std::shared_ptr<class PTTexture> m_EmissiveTexture;
    std::shared_ptr<class PTTexture> m_MetalRoughnessTexture;
    std::shared_ptr<class PTTexture> m_NormalTexture;

    // A flag to keep track if the GPU representation of the material needs to be updated after something was changed
    mutable bool m_DeviceMaterialDirty;
    // A small GPU memory buffer to store the GPU representation in
    mutable std::unique_ptr<class MemoryBuffer> m_DeviceMemoryBuffer;
};