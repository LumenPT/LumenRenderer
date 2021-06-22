#include "PTMaterial.h"


#include "MemoryBuffer.h"
#include "../Shaders/CppCommon/ModelStructs.h"
#include "PTTexture.h"

#include "Cuda/vector_functions.h"

PTMaterial::PTMaterial()
    : m_MaterialData(0.f),
      m_DeviceMaterialDirty(true)
{
    // Allocate the GPU memory for the GPU material representation
    m_DeviceMemoryBuffer = std::make_unique<MemoryBuffer>(sizeof(DeviceMaterial));

	//Set the roughness to 1 by default.
    m_MaterialData.SetRoughness(1.f);
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
    m_MaterialData.SetColor(make_float4(a_NewDiffuseColor.x, a_NewDiffuseColor.y, a_NewDiffuseColor.z, a_NewDiffuseColor.w));
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetDiffuseTexture(std::shared_ptr<Lumen::ILumenTexture> a_NewDiffuseTexture)
{
    m_DiffuseTexture = *reinterpret_cast<std::shared_ptr<PTTexture>*>(&a_NewDiffuseTexture);
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetEmission(const glm::vec3& a_EmissiveVal)
{
    m_MaterialData.SetEmissive(make_float3(a_EmissiveVal.x, a_EmissiveVal.y, a_EmissiveVal.z));
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
    const auto diffuseColor = m_MaterialData.GetColor();
    return glm::vec4(diffuseColor.x, diffuseColor.y, diffuseColor.z, diffuseColor.w);
}

glm::vec3 PTMaterial::GetEmissiveColor() const
{
    const auto emissiveColor = m_MaterialData.GetEmissive();
    return glm::vec3(emissiveColor.x, emissiveColor.y, emissiveColor.z);
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

    if(m_TransmissionTexture)
    {
        m.m_TransmissionTexture = **m_TransmissionTexture;
    }

    if(m_ClearCoatTexture)
    {
        m.m_ClearCoatTexture = **m_ClearCoatTexture;
    }

    if(m_ClearCoatRoughnessTexture)
    {
        m.m_ClearCoatTexture = **m_ClearCoatRoughnessTexture;
    }

    if (m_TintTexture)
    {
        m.m_TintTexture = **m_TintTexture;
    }

	//Ensure all textures are present, as defaults are required.
    assert(m_TintTexture);
    assert(m_ClearCoatRoughnessTexture);
    assert(m_ClearCoatTexture);
    assert(m_TransmissionTexture);
    assert(m_DiffuseTexture);
    assert(m_EmissiveTexture);
    assert(m_NormalTexture);
    assert(m_MetalRoughnessTexture);

    //Set the tightly packed properties.
    m.m_MaterialData = m_MaterialData;
	
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
    m_MaterialData.SetClearCoat(a_Factor);
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetClearCoatRoughnessFactor(float a_Factor)
{
	//Invert because our shading uses gloss instead of roughness.
    m_MaterialData.SetClearCoatGloss(1.f - a_Factor);
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetIndexOfRefraction(float a_Factor)
{
    m_MaterialData.SetRefractiveIndex(a_Factor);
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetTransmissionFactor(float a_Factor)
{
    m_MaterialData.SetTransmission(a_Factor);
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetSpecularFactor(float a_Factor)
{
    m_MaterialData.SetSpecular(a_Factor);
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetSpecularTintFactor(float a_Factor)
{
    m_MaterialData.SetSpecTint(a_Factor);
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetSubSurfaceFactor(float a_Factor)
{
    m_MaterialData.SetSubSurface(a_Factor);
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetLuminance(float a_Factor)
{
    m_MaterialData.SetLuminance(a_Factor);
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetTintTexture(std::shared_ptr<Lumen::ILumenTexture> a_Texture)
{
    m_TintTexture = *reinterpret_cast<std::shared_ptr<PTTexture>*>(&a_Texture);
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetTintFactor(const glm::vec3& a_Factor)
{
    m_MaterialData.SetTint(make_float3(a_Factor.x, a_Factor.y, a_Factor.z));
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetAnisotropic(float a_Factor)
{
    m_MaterialData.SetAnisotropic(a_Factor);
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetSheenFactor(float a_Factor)
{
    m_MaterialData.SetSheen(a_Factor);
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetSheenTintFactor(float a_Factor)
{
    m_MaterialData.SetSheenTint(a_Factor);
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetTransmittanceFactor(const glm::vec3& a_Factor)
{
    m_MaterialData.SetTransmittance(make_float3(a_Factor.x, a_Factor.y, a_Factor.z));
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetMetallicFactor(float a_Factor)
{
    m_MaterialData.SetMetallic(a_Factor);
    m_DeviceMaterialDirty = true;
}

void PTMaterial::SetRoughnessFactor(float a_Factor)
{
    assert(a_Factor > 0.f && a_Factor <= 1.f && "Roughness can't be 0. ");
    m_MaterialData.SetRoughness(a_Factor);
    m_DeviceMaterialDirty = true;
}
