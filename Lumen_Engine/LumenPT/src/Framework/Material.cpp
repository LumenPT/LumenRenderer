#include "Material.h"


#include "MemoryBuffer.h"
#include "../Shaders/CppCommon/ModelStructs.h"
#include "Texture.h"

#include "Cuda/vector_functions.h"

Material::Material()
    : m_DeviceMaterialDirty(true)
{
    m_DeviceMemoryBuffer = std::make_unique<MemoryBuffer>(sizeof(DeviceMaterial));
}

DeviceMaterial* Material::GetDeviceMaterial() const
{
    if (m_DeviceMaterialDirty)
    {
        m_DeviceMaterialDirty = false;

        auto devMat = CreateDeviceMaterial();

        m_DeviceMemoryBuffer->Write(devMat);
    }

    return m_DeviceMemoryBuffer->GetDevicePtr<DeviceMaterial>();
}

void Material::SetDiffuseColor(const glm::vec4& a_NewDiffuseColor)
{
    m_DiffuseColor = make_float4(a_NewDiffuseColor.x, a_NewDiffuseColor.y, a_NewDiffuseColor.z, a_NewDiffuseColor.w);
    m_DeviceMaterialDirty = true;
}

void Material::SetDiffuseTexture(std::shared_ptr<Lumen::ILumenTexture> a_NewDiffuseTexture)
{
    m_DiffuseTexture = *reinterpret_cast<std::shared_ptr<Texture>*>(&a_NewDiffuseTexture);
    m_DeviceMaterialDirty = true;
}

void Material::SetEmission(const glm::vec3& a_EmissiveVal)
{
    m_EmissiveColor = make_float3(a_EmissiveVal.x, a_EmissiveVal.y, a_EmissiveVal.z);
    m_DeviceMaterialDirty = true;
}

void Material::SetEmissiveTexture(std::shared_ptr<Lumen::ILumenTexture> a_EmissiveTexture)
{
	
}

glm::vec4 Material::GetDiffuseColor() const
{
    return glm::vec4(m_DiffuseColor.x, m_DiffuseColor.y, m_DiffuseColor.z, m_DiffuseColor.w);
}

Lumen::ILumenTexture& Material::GetDiffuseTexture() const
{
    return *m_DiffuseTexture;
}

DeviceMaterial Material::CreateDeviceMaterial() const
{
    DeviceMaterial m;
    m.m_DiffuseColor = m_DiffuseColor;
    m.m_EmissionColor = m_EmissiveColor;
    if (m_DiffuseTexture)
    {
        m.m_DiffuseTexture = **m_DiffuseTexture;
    }

    return m;
}
