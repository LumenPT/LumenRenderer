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
    , m_SelectedNode(nullptr)
    , m_DisplayNodes(true)
{
    m_SearchString.resize(128, 0);
}

void Lumen::SceneGraph::Display(ILumenScene& a_Scene)
{


    ImGui::Begin("Scene Graph");

    if (ImGui::Button("Node View"))
        m_DisplayNodes = true;
    ImGui::SameLine();
    if (ImGui::Button("Instance View"))
        m_DisplayNodes = false;

    if (ImGui::Button("Add Volume"))
        a_Scene.AddVolume();

    if (m_DisplayNodes)
    {
        NodeSelection(a_Scene);
    }
    else
    {
        InstanceSelection(a_Scene);
    }

    if (m_SelectedMeshInstance || m_SelectedVolumeInstance)
    {
        if (ImGui::Button("Remove"))
        {
            if (m_SelectedMeshInstance)
            {
                a_Scene.m_MeshInstances.erase(std::find_if(a_Scene.m_MeshInstances.begin(), a_Scene.m_MeshInstances.end(),
                    [this](std::unique_ptr<MeshInstance>& a_Ptr) {
                        return m_SelectedMeshInstance == a_Ptr.get();
                    }));
                m_SelectedMeshInstance = nullptr;
            }
            else
            {
                a_Scene.m_VolumeInstances.erase(std::find_if(a_Scene.m_VolumeInstances.begin(), a_Scene.m_VolumeInstances.end(),
                    [this](std::unique_ptr<VolumeInstance>& a_Ptr) {
                        return m_SelectedVolumeInstance == a_Ptr.get();
                    }));
                m_SelectedVolumeInstance = nullptr;
            }
        }
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
        ImGui::Separator();
        auto density = m_SelectedVolumeInstance->GetDensity();
        auto newDensity = density;
        ImGui::DragFloat("Volume Density", &newDensity, 0.005f, 0.0f, 1.0f);
        if (newDensity != density)
        {
            m_SelectedVolumeInstance->SetDensity(newDensity);
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

    ImGui::Text("Transform ID %llu", a_Transform.m_ID);
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

struct DragDrop
{
    Lumen::ILumenScene::Node* m_Source;
    Lumen::ILumenScene::Node* m_Dest;
    Lumen::ILumenScene::Node* m_Target;
};

void Lumen::SceneGraph::NodeSelection(Lumen::ILumenScene& a_Scene)
{
    if (ImGui::ListBoxHeader("", ImVec2(m_GraphSize.x, m_GraphSize.y)))
    {
        std::map<std::string, uint32_t> names;

        for (auto& m_RootNode : a_Scene.m_RootNodes)
        {
            DisplayNode(*m_RootNode, names);
        }
        ImGui::ListBoxFooter();
    }

    if (m_SelectedNode)
    {
        TransformEditor(m_SelectedNode->m_Transform);
    }
}

void Lumen::SceneGraph::InstanceSelection(Lumen::ILumenScene& a_Scene)
{
    ImGui::InputText("Search", m_SearchString.data(), m_SearchString.size());

    auto trimmed = std::string(m_SearchString.begin(), m_SearchString.begin() + m_SearchString.find(char(0)));

    for (auto& c : trimmed)
    {
        c = std::tolower(c);
    }

    if (ImGui::ListBoxHeader("", ImVec2(m_GraphSize.x, m_GraphSize.y)))
    {
        std::map<std::string, uint32_t> names;

        for (auto& meshInstance : a_Scene.m_MeshInstances)
        {
            auto instanceName = "[M] " + meshInstance->m_Name;
            if (names[meshInstance->m_Name] == 0)
                names[meshInstance->m_Name]++;
            else
                instanceName += std::to_string(names[meshInstance->m_Name]);

            std::string lowerCase;
            lowerCase.reserve(instanceName.size());

            for (char& c : instanceName)
            {
                lowerCase.push_back(std::tolower(c));
            }

            if (trimmed.empty() || lowerCase.find(trimmed) != std::string::npos)
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
}

void Lumen::SceneGraph::DisplayNode(Lumen::ILumenScene::Node& a_Node, std::map<std::string, uint32_t>& a_NameMap, uint16_t a_Depth)
{
    std::string name = a_Node.m_Name;

    if (name.empty())
        name = "Unnamed Node";

    if (a_NameMap[name])
        name += ' ' + std::to_string(a_NameMap[a_Node.m_Name]);
    a_NameMap[a_Node.m_Name]++;

    int flags = ImGuiTreeNodeFlags_OpenOnArrow;
    flags |= ImGuiTreeNodeFlags_OpenOnDoubleClick;

    bool collapsed = true;

    if (!a_Node.m_ChildNodes.empty())
    {
        if (m_SelectedNode == &a_Node)
            ImGui::PushStyleColor(ImGuiCol_Header, ImGui::GetColorU32(ImGuiCol_ButtonHovered));
        else
            ImGui::PushStyleColor(ImGuiCol_Header, ImGui::GetColorU32(ImGuiCol_ChildBg));
        if (ImGui::CollapsingHeader(name.c_str(), flags))
            collapsed = false;
    }
    else
    {
        if (a_Node.m_MeshInstancePtr == m_SelectedMeshInstance)
            ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetColorU32(ImGuiCol_ButtonHovered));
        else
            ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetColorU32(ImGuiCol_ChildBg));
        if (ImGui::Selectable(name.c_str()))
        {
            m_SelectedMeshInstance = a_Node.m_MeshInstancePtr;
            m_SelectedNode = nullptr;
        }
    }

    ImGui::PopStyleColor();

    DragDrop dd = {};

    if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceNoDisableHover))
    {
        dd.m_Source = a_Node.m_Parent;
        dd.m_Target = &a_Node;
        ImGui::Text(name.c_str());
        ImGui::SetDragDropPayload("Node_DND", &dd, sizeof(dd));
        ImGui::EndDragDropSource();
    }

    //if (ImGui::BeginDragDropTarget())
    //{
    //    ImGuiDragDropFlags target_flags = 0;
    //    target_flags |= ImGuiDragDropFlags_AcceptBeforeDelivery;    // Don't wait until the delivery (release mouse button on a target) to do something
    //    target_flags |= ImGuiDragDropFlags_AcceptNoDrawDefaultRect; // Don't display the yellow rectangle
    //    if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("DND_DEMO_NAME", target_flags))
    //    {
    //        move_from = *(const int*)payload->Data;
    //        move_to = n;
    //    }
    //    ImGui::EndDragDropTarget();
    //}

    if (ImGui::BeginDragDropTarget())
    {
        ImGuiDragDropFlags target_flags = 0;
        //target_flags |= ImGuiDragDropFlags_AcceptBeforeDelivery;    // Don't wait until the delivery (release mouse button on a target) to do something
        //target_flags |= ImGuiDragDropFlags_AcceptNoDrawDefaultRect; // Don't display the yellow rectangle
        auto payload = ImGui::AcceptDragDropPayload("Node_DND", target_flags);

        if (payload)
        {
            auto data = reinterpret_cast<const DragDrop*>(payload->Data);
            dd.m_Target = data->m_Target;
            dd.m_Source = data->m_Source;
            dd.m_Dest = &a_Node;
        }
        ImGui::EndDragDropTarget();
    }

    if (dd.m_Dest && dd.m_Target)
    {
        auto trg = dd.m_Target;
        ILumenScene::Node* toMove = nullptr;
        if (dd.m_Source)
        {
            for (size_t i = 0; i < dd.m_Source->m_ChildNodes.size(); i++)
            {
                if (dd.m_Source->m_ChildNodes[i].get() == trg)
                {
                    toMove = dd.m_Source->m_ChildNodes[i].release(); // Manually move the pointer if it exists
                    dd.m_Source->m_ChildNodes.erase(dd.m_Source->m_ChildNodes.begin() + i);
                    break;
                }
            }
        }
        else
            toMove = dd.m_Target;

        auto destIsChildOfSrc = dd.m_Dest->IsChildOf(*dd.m_Target);
        if (!destIsChildOfSrc)
        {
            dd.m_Dest->m_ChildNodes.push_back(std::unique_ptr<ILumenScene::Node>(toMove));
            dd.m_Dest->m_Transform.AddChild(dd.m_Target->m_Transform);
            dd.m_Target->m_Parent = dd.m_Dest;

            printf("Parent node changed\n");
        }
    }

    if (!a_Node.m_ChildNodes.empty() && ImGui::IsItemClicked())
    {
        m_SelectedNode = &a_Node;
        m_SelectedMeshInstance = nullptr;
        m_SelectedVolumeInstance = nullptr;
    }
    if (!collapsed)
    {
        ImGui::Indent();
        for (size_t i = 0; i < a_Node.m_ChildNodes.size(); i++)
        {
            auto& child = a_Node.m_ChildNodes[i];
            DisplayNode(*child, a_NameMap, a_Depth + 1);
        }
        ImGui::Unindent();
        //ImGui::TreePop();
    }

}