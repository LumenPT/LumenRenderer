#include "PTMaterial.h"


#include "MemoryBuffer.h"
#include "../Shaders/CppCommon/ModelStructs.h"
#include "PTTexture.h"

#include "Cuda/vector_functions.h"

PTMaterial::PTMaterial()
    : m_DeviceMaterialDirty(true)
{
    // Allocate the GPU memory for the GPU material representation
    m_DeviceMemoryBuffer = std::make_unique<MemoryBuffer>(sizeof(DeviceMaterial));
}

DeviceMaterial* PTMaterial::GetDeviceMaterial() const
{
    // Update the GPU material if the dirty flag is raised
    if (m_DeviceMaterialDirty)
    {
        m_DeviceMaterialDirty = false;

        // Create a new material using the CPU to GPU conversion function
        auto devMat = CreateDeviceMaterial();

        // Write the new material to the pre-allocated GPU memory buffer
        m_DeviceMemoryBuffer->Write(devMat);
    }

    return m_DeviceMemoryBuffer->GetDevicePtr<DeviceMaterial>();
}

void PTMaterial::SetDiffuseColor(const glm::vec4& a_NewDiffuseColor)
{
    m_DiffuseColor = make_float4(a_NewDiffuseColor.x, a_NewDiffuseColor.y, a_NewDiffuseColor.z, a_NewDiffuseColor.w);
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetDiffuseTexture(std::shared_ptr<Lumen::ILumenTexture> a_NewDiffuseTexture)
{
    m_DiffuseTexture = *reinterpret_cast<std::shared_ptr<PTTexture>*>(&a_NewDiffuseTexture);
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetEmission(const glm::vec3& a_EmissiveVal)
{
    m_EmissiveColor = make_float3(a_EmissiveVal.x, a_EmissiveVal.y, a_EmissiveVal.z);
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetEmissiveTexture(std::shared_ptr<Lumen::ILumenTexture> a_EmissiveTexture)
{
	m_EmissiveTexture = *reinterpret_cast<std::shared_ptr<PTTexture>*>(&a_EmissiveTexture);
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetMetalRoughnessTexture(std::shared_ptr<Lumen::ILumenTexture> a_MetalRoughnessTexture)
{
    m_MetalRoughnessTexture = *reinterpret_cast<std::shared_ptr<PTTexture>*>(&a_MetalRoughnessTexture);
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetNormalTexture(std::shared_ptr<Lumen::ILumenTexture> a_NormalTexture)
{
    m_NormalTexture = *reinterpret_cast<std::shared_ptr<PTTexture>*>(&a_NormalTexture);
    m_DeviceMaterialDirty = true;
}

glm::vec4 PTMaterial::GetDiffuseColor() const
{
    return glm::vec4(m_DiffuseColor.x, m_DiffuseColor.y, m_DiffuseColor.z, m_DiffuseColor.w);
}

Lumen::ILumenTexture& PTMaterial::GetDiffuseTexture() const
{
    return *m_DiffuseTexture;
}

DeviceMaterial PTMaterial::CreateDeviceMaterial() const
{
    DeviceMaterial m;
    m.m_DiffuseColor = m_DiffuseColor;
    m.m_EmissionColor = m_EmissiveColor;

    //Should always have a default loaded.
    m.m_MetalRoughnessTexture = **m_MetalRoughnessTexture;

    //Should always be default loaded
    m.m_NormalTexture = **m_NormalTexture;


    if (m_DiffuseTexture)
    {
        m.m_DiffuseTexture = **m_DiffuseTexture;
    }

    return m;
}
