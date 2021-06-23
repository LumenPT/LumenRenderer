#include "lmnpch.h"
#include "SceneGraph.h"

#include "../ModelLoading/ILumenScene.h"
#include "../Renderer/LumenRenderer.h"

#include "../ImGui/imgui.h"

#include <map>
#include <string>

Lumen::SceneGraph::SceneGraph()
	: m_SelectedMeshInstance(nullptr)
	, m_SelectedVolumeInstance(nullptr)
{
    m_SearchString.resize(128, 0);
}

void Lumen::SceneGraph::Display(ILumenScene& a_Scene)
{

    ImGui::Begin("Scene Graph");
    ImGui::InputText("Search", m_SearchString.data(), m_SearchString.size());

    auto trimmed = std::string(m_SearchString.begin(), m_SearchString.begin() + m_SearchString.find(char(0)));
	
    if (ImGui::ListBoxHeader(""))
    {
        std::map<std::string, uint32_t> names;
    	
        for (auto& meshInstance : a_Scene.m_MeshInstances)
        {
            auto instanceName = "[M] " + meshInstance->m_Name;
            if (names[meshInstance->m_Name] == 0)
                names[meshInstance->m_Name]++;
            else
                instanceName += std::to_string(names[meshInstance->m_Name]);

            if (trimmed.empty() || instanceName.find(trimmed) != std::string::npos)
                if (ImGui::Selectable(instanceName.c_str(), m_SelectedMeshInstance == meshInstance.get()))
                {
                    m_SelectedMeshInstance = meshInstance.get();
                    m_SelectedVolumeInstance = nullptr;
                }
        }

        for (auto& volumeInstance : a_Scene.m_VolumeInstances)
        {
            auto instanceName = "[V] " + volumeInstance->m_Name;
            if (names[volumeInstance->m_Name] = 0)
                names[volumeInstance->m_Name]++;
            else
                instanceName += std::to_string(names[volumeInstance->m_Name]);

            if (trimmed.empty() || instanceName.find(trimmed) != std::string::npos)
                if (ImGui::Selectable(instanceName.c_str(), m_SelectedVolumeInstance == volumeInstance.get()))
                {
                    m_SelectedVolumeInstance = volumeInstance.get();
                    m_SelectedMeshInstance = nullptr;

                }
        }
    	
        ImGui::ListBoxFooter();
    }

    if (m_SelectedMeshInstance != nullptr)
    {
        if (ImGui::CollapsingHeader("Transformation"))
        {
            auto& transform = m_SelectedMeshInstance->m_Transform;
            TransformEditor(transform);            
        }

        if (ImGui::CollapsingHeader("Emissive Properties"))
        {
            auto emissiveness = m_SelectedMeshInstance->GetEmissiveness();
            auto newEmissiveness = emissiveness;
            ImGui::Text("Emissive Properties");

            bool overrideEmissiveness = newEmissiveness.m_EmissionMode == EmissionMode::OVERRIDE;

            ImGui::Checkbox("Override Emissiveness", &overrideEmissiveness);
            if (overrideEmissiveness)
                newEmissiveness.m_EmissionMode = EmissionMode::OVERRIDE;
            else
                newEmissiveness.m_EmissionMode = EmissionMode::ENABLED;

            ImGui::ColorEdit3("Emission Factor", &newEmissiveness.m_OverrideRadiance[0], ImGuiColorEditFlags_Float);
            ImGui::DragFloat("Emission Scale", &newEmissiveness.m_Scale, 1, 0.001f, std::numeric_limits<float>::max());

            if (newEmissiveness.m_Scale != emissiveness.m_Scale ||
                newEmissiveness.m_EmissionMode != emissiveness.m_EmissionMode ||
                newEmissiveness.m_OverrideRadiance != emissiveness.m_OverrideRadiance)
            {
                m_SelectedMeshInstance->SetEmissiveness(newEmissiveness);            
            }            
        }

        if (ImGui::CollapsingHeader("Material Properties"))
        {
            if (m_SelectedMeshInstance->GetOverrideMaterial() == nullptr)
            {
                ImGui::Text("This mesh instance has no override material specified.");
                if (ImGui::Button("Add Override Material"))
                {
                    auto newMat = m_RendererRef->CreateMaterial();
                    m_SelectedMeshInstance->SetOverrideMaterial(newMat);
                }
            }
            else
            {
                // Steel thy souls, for pain shall be upon thee
                auto mat = m_SelectedMeshInstance->GetOverrideMaterial();
                auto colorFactor = mat->GetDiffuseColor();
                auto clearCoatFactor = mat->GetClearCoatFactor();
                auto clearCoatRoughnessFactor = mat->GetClearCoatRoughnessFactor();
                auto luminance = mat->GetLuminance();
                auto sheenFactor = mat->GetSheenFactor();
                auto sheenTintFactor = mat->GetSheenTintFactor();
                auto anisotropic = mat->GetAnisotropic();
                auto tintFactor = mat->GetTintFactor();
                auto transmissionFactor = mat->GetTransmissionFactor();
                auto transmittanceFactor = mat->GetTransmittanceFactor();
                auto indexOfRefraction = mat->GetIndexOfRefraction();
                auto specularFactor = mat->GetSpecularFactor();
                auto specularTintFactor = mat->GetSpecularTintFactor();
                auto subsurfaceFactor = mat->GetSubSurfaceFactor();
                auto metallicFactor = mat->GetMetallicFactor();
                auto roughnessFactor = mat->GetRoughnessFactor();

                auto newColorFactor = mat->GetDiffuseColor();
                auto newClearCoatFactor = mat->GetClearCoatFactor();
                auto newClearCoatRoughnessFactor = mat->GetClearCoatRoughnessFactor();
                auto newLuminance = mat->GetLuminance();
                auto newSheenFactor = mat->GetSheenFactor();
                auto newSheenTintFactor = mat->GetSheenTintFactor();
                auto newAnisotropic = mat->GetAnisotropic();
                auto newTintFactor = mat->GetTintFactor();
                auto newTransmissionFactor = mat->GetTransmissionFactor();
                auto newTransmittanceFactor = mat->GetTransmittanceFactor();
                auto newIndexOfRefraction = mat->GetIndexOfRefraction();
                auto newSpecularFactor = mat->GetSpecularFactor();
                auto newSpecularTintFactor = mat->GetSpecularTintFactor();
                auto newSubsurfaceFactor = mat->GetSubSurfaceFactor();
                auto newMetallicFactor = mat->GetMetallicFactor();
                auto newRoughnessFactor = mat->GetRoughnessFactor();

                ImGui::DragFloat3("Diffuse Color Factor", &newColorFactor[0]);
                ImGui::DragFloat("Clear Coat Factor", &newClearCoatFactor);
                ImGui::DragFloat("Clear Coat Roughness Factor", &newClearCoatRoughnessFactor);
                ImGui::DragFloat("Luminance", &newLuminance);
                ImGui::DragFloat("Sheen Factor", &newSheenFactor);
                ImGui::DragFloat("Sheen Tint Factor", &newSheenTintFactor);
                ImGui::DragFloat("Anisotropic", &newAnisotropic);
                ImGui::DragFloat3("Tint Factor", &newTintFactor[0]);
                ImGui::DragFloat("Transmission Factor", &newTransmissionFactor);
                ImGui::DragFloat3("Transmittance Factor", &newTransmittanceFactor[0]);
                ImGui::DragFloat("Index of Refraction", &newIndexOfRefraction);
                ImGui::DragFloat("Specular Factor", &newSpecularFactor);
                ImGui::DragFloat("Specular Tint Factor", &newSpecularTintFactor);
                ImGui::DragFloat("Subsurface Factor", &newSubsurfaceFactor);
                ImGui::DragFloat("Metallic Factor", &newMetallicFactor);
                ImGui::DragFloat("Roughness Factor", &newRoughnessFactor);

                bool matChanged = false;

                if (newColorFactor != colorFactor)
                {
                    mat->SetDiffuseColor(newColorFactor);
                    matChanged = true;
                }
                if (newClearCoatFactor != clearCoatFactor)
                {
                    mat->SetClearCoatFactor(newClearCoatFactor);
                    matChanged = true;
                }
                if (newClearCoatRoughnessFactor != clearCoatRoughnessFactor)
                {
                    mat->SetClearCoatRoughnessFactor(newClearCoatRoughnessFactor);
                    matChanged = true;
                }
                if (newLuminance != luminance)
                {
                    mat->SetLuminance(newLuminance);
                    matChanged = true;
                }
                if (newSheenFactor != sheenFactor)
                {
                    mat->SetSheenFactor(newSheenFactor);
                    matChanged = true;
                }
                if (newSheenTintFactor != sheenTintFactor)
                {
                    mat->SetSheenTintFactor(newSheenTintFactor);
                    matChanged = true;
                }
                if (newAnisotropic != anisotropic)
                {
                    mat->SetAnisotropic(newAnisotropic);
                    matChanged = true;
                }
                if (newTintFactor != tintFactor)
                {
                    mat->SetTintFactor(newTintFactor);
                    matChanged = true;
                }
                if (newTransmissionFactor != transmissionFactor)
                {
                    mat->SetTransmissionFactor(newTransmissionFactor);
                    matChanged = true;
                }
                if (newTransmittanceFactor != transmittanceFactor)
                {
                    mat->SetTransmittanceFactor(newTransmittanceFactor);
                    matChanged = true;
                }
                if (newIndexOfRefraction != indexOfRefraction)
                {
                    mat->SetIndexOfRefraction(newIndexOfRefraction);
                    matChanged = true;
                }
                if (newSpecularFactor != specularFactor)
                {
                    mat->SetSpecularFactor(newSpecularFactor);
                    matChanged = true;
                }
                if (newSpecularTintFactor != specularTintFactor)
                {
                    mat->SetSpecularTintFactor(newSpecularTintFactor);
                    matChanged = true;
                }
                if (newSubsurfaceFactor != subsurfaceFactor)
                {
                    mat->SetSubSurfaceFactor(newSubsurfaceFactor);
                    matChanged = true;
                }
                if (newMetallicFactor != metallicFactor)
                {
                    mat->SetMetallicFactor(newMetallicFactor);
                    matChanged = true;
                }
                if (newRoughnessFactor != roughnessFactor)
                {
                    mat->SetRoughnessFactor(newRoughnessFactor);
                    matChanged = true;
                }

                if (matChanged)
                {
                    m_SelectedMeshInstance->SetOverrideMaterial(mat);
                }
            }
        }

    }
    else if (m_SelectedVolumeInstance != nullptr)
    {
	    if (ImGui::CollapsingHeader("Transformation"))
        {
            auto& transform = m_SelectedVolumeInstance->m_Transform;
            TransformEditor(transform);            
        }
    }
    ImGui::End();
}

void Lumen::SceneGraph::TransformEditor(Transform& a_Transform)
{
    glm::vec3 t, r, s;
    t = a_Transform.GetPosition();
    r = a_Transform.GetRotationEuler();
    s = a_Transform.GetScale();

    ImGui::DragFloat3("Position", &t[0]);
    ImGui::DragFloat3("Rotation", &r[0]);
    ImGui::DragFloat3("Scale", &s[0]);

    if (t != a_Transform.GetPosition())
        a_Transform.SetPosition(t);

    if (s != a_Transform.GetScale())
        a_Transform.SetScale(s);

    auto deltaR = r - a_Transform.GetRotationEuler();
    if (glm::length(deltaR) != 0)
    {
        glm::quat deltaRQuat(glm::radians(deltaR));
        a_Transform.Rotate(deltaRQuat);
    }
}

