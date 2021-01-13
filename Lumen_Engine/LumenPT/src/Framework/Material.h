#pragma once

#include "Renderer/ILumenResources.h"

#include "glm/vec4.hpp"

#include "Cuda/vector_types.h"

#include <memory>

struct DeviceMaterial;

class Material : public Lumen::ILumenMaterial
{
public:
    Material();

    DeviceMaterial* GetDeviceMaterial() const;

    void SetDiffuseColor(const glm::vec4& a_NewDiffuseColor) override;
    void SetDiffuseTexture(std::shared_ptr<Lumen::ILumenTexture> a_NewDiffuseTexture) override;

    glm::vec4 GetDiffuseColor() const override;
    Lumen::ILumenTexture& GetDiffuseTexture() const override;

private:

    DeviceMaterial CreateDeviceMaterial() const;

    float4 m_DiffuseColor;
    std::shared_ptr<class Texture> m_DiffuseTexture;

    mutable bool m_DeviceMaterialDirty;
    mutable std::unique_ptr<class MemoryBuffer> m_DeviceMemoryBuffer;
};