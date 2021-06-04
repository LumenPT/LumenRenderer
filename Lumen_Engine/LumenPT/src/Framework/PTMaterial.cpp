#include "PTMaterial.h"


#include "MemoryBuffer.h"
#include "../Shaders/CppCommon/ModelStructs.h"
#include "PTTexture.h"

#include "Cuda/vector_functions.h"

PTMaterial::PTMaterial()
    : m_DiffuseColor(), m_EmissiveColor(), m_TransmissionFactor(0), m_ClearCoatFactor(0), m_ClearCoatRoughnessFactor(0),
      m_IndexOfRefraction(1.f),
      m_SpecularFactor(0),
      m_SpecularTintFactor(0),
      m_SubSurfaceFactor(0), m_Luminance(0), m_Anisotropic(0),
      m_SheenFactor(0.f), m_SheenTintFactor(0), m_MetallicFactor(1.f), m_RoughnessFactor(1.f),
      m_TintFactor(make_float3(0.f)), m_Transmittance(),
      m_DeviceMaterialDirty(true)
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
    m_EmissiveColor = make_float4(a_EmissiveVal.x, a_EmissiveVal.y, a_EmissiveVal.z, 1.f);
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

glm::vec3 PTMaterial::GetEmissiveColor() const
{
    return glm::vec3(m_EmissiveColor.x, m_EmissiveColor.y, m_EmissiveColor.z);
}


Lumen::ILumenTexture& PTMaterial::GetDiffuseTexture() const
{
    return *m_DiffuseTexture;
}

Lumen::ILumenTexture& PTMaterial::GetEmissiveTexture() const
{
    return *m_EmissiveTexture;
}

DeviceMaterial PTMaterial::CreateDeviceMaterial() const
{
    DeviceMaterial m;
    m.m_DiffuseColor = m_DiffuseColor;
    m.m_EmissionColor = m_EmissiveColor;

    //Should always have a default loaded.
    if (m_MetalRoughnessTexture)
        m.m_MetalRoughnessTexture = **m_MetalRoughnessTexture;

    //Should always be default loaded
    if (m_NormalTexture)
        m.m_NormalTexture = **m_NormalTexture;

    if (m_EmissiveTexture)
        m.m_EmissiveTexture = **m_EmissiveTexture;


    if (m_DiffuseTexture)
    {
        m.m_DiffuseTexture = **m_DiffuseTexture;
    }


    //Disney BSDF stuff
    m.m_TransmissionFactor = m_TransmissionFactor;
    m.m_ClearCoatFactor = m_ClearCoatFactor;
    m.m_IndexOfRefraction = m_IndexOfRefraction;
    m.m_ClearCoatRoughnessFactor = m_ClearCoatRoughnessFactor;
    m.m_SpecularFactor = m_SpecularFactor;
    m.m_SpecularTintFactor = m_SpecularTintFactor;
    m.m_SubSurfaceFactor = m_SubSurfaceFactor;

    if(m_TransmissionTexture)
    {
        m.m_TransmissionFactor = **m_TransmissionTexture;
    }

    if(m_ClearCoatTexture)
    {
        m.m_ClearCoatTexture = **m_ClearCoatTexture;
    }

    if(m_ClearCoatRoughnessTexture)
    {
        m.m_ClearCoatRoughnessFactor = **m_ClearCoatRoughnessTexture;
    }

    if (m_TintTexture)
    {
        m.m_TintTexture = **m_TintTexture;
    }

    m.m_Luminance = m_Luminance;
    m.m_SheenFactor = m_SheenFactor;
    m.m_SheenTintFactor = m_SheenTintFactor;
    m.m_TintFactor = m_TintFactor;
    m.m_Anisotropic = m_Anisotropic;
    m.m_TransmittanceFactor = m_Transmittance;

    m.m_MetallicFactor = m_MetallicFactor;
    m.m_RoughnessFactor = m_RoughnessFactor;

    return m;
}

void PTMaterial::SetClearCoatTexture(std::shared_ptr<Lumen::ILumenTexture> a_Texture)
{
    m_ClearCoatTexture = *reinterpret_cast<std::shared_ptr<PTTexture>*>(&a_Texture);
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetClearCoatRoughnessTexture(std::shared_ptr<Lumen::ILumenTexture> a_Texture)
{
    m_ClearCoatRoughnessTexture = *reinterpret_cast<std::shared_ptr<PTTexture>*>(&a_Texture);
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetTransmissionTexture(std::shared_ptr<Lumen::ILumenTexture> a_Texture)
{
    m_TransmissionTexture = *reinterpret_cast<std::shared_ptr<PTTexture>*>(&a_Texture);
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetClearCoatFactor(float a_Factor)
{
    m_ClearCoatFactor = a_Factor;
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetClearCoatRoughnessFactor(float a_Factor)
{
    m_ClearCoatRoughnessFactor = a_Factor;
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetIndexOfRefraction(float a_Factor)
{
    m_IndexOfRefraction = a_Factor;
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetTransmissionFactor(float a_Factor)
{
    m_TransmissionFactor = a_Factor;
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetSpecularFactor(float a_Factor)
{
    m_SpecularFactor = a_Factor;
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetSpecularTintFactor(float a_Factor)
{
    m_SpecularTintFactor = a_Factor;
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetSubSurfaceFactor(float a_Factor)
{
    m_SubSurfaceFactor = a_Factor;
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetLuminance(float a_Factor)
{
    m_Luminance = a_Factor;
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetTintTexture(std::shared_ptr<Lumen::ILumenTexture> a_Texture)
{
    m_TintTexture = *reinterpret_cast<std::shared_ptr<PTTexture>*>(&a_Texture);
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetTintFactor(const glm::vec3& a_Factor)
{
    m_TintFactor = make_float3(a_Factor.x, a_Factor.y, a_Factor.z);
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetAnisotropic(float a_Factor)
{
    m_Anisotropic = a_Factor;
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetSheenFactor(float a_Factor)
{
    m_SheenFactor = a_Factor;
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetSheenTintFactor(float a_Factor)
{
    m_SheenTintFactor = a_Factor;
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetTransmittanceFactor(const glm::vec3& a_Factor)
{
    m_Transmittance = make_float3(a_Factor.x, a_Factor.y, a_Factor.z);
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetMetallicFactor(float a_Factor)
{
    m_MetallicFactor = a_Factor;
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetRoughnessFactor(float a_Factor)
{
    m_RoughnessFactor = a_Factor;
    m_DeviceMaterialDirty = true;
}
