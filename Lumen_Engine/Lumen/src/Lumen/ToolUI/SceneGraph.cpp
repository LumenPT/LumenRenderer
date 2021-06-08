#include "lmnpch.h"
#include "SceneGraph.h"

#include "../ModelLoading/ILumenScene.h"

#include "../ImGui/imgui.h"

void Lumen::SceneGraph::Display(ILumenScene& a_Scene)
{
    ImGui::Begin("Scene Graph");

    if (ImGui::ListBoxHeader(""))
    {
        for (auto& meshInstance : a_Scene.m_MeshInstances)
        {
            if (ImGui::Selectable(meshInstance->m_Name.c_str(), m_SelectedMeshInstance == meshInstance.get()))
                m_SelectedMeshInstance = meshInstance.get();
        }
        ImGui::ListBoxFooter();
    }

    if (m_SelectedMeshInstance != nullptr)
    {
        auto& transform = m_SelectedMeshInstance->m_Transform;
        TransformEditor(transform);


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

    auto deltaR = a_Transform.GetRotationEuler() - r;
    if (glm::length(deltaR) != 0)
    {
        glm::quat deltaRQuat(deltaR);
        a_Transform.Rotate(deltaRQuat);
    }
}

