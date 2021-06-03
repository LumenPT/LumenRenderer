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
        m_SelectedMeshInstance;
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

