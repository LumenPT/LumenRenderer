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
            if (m_SelectedMeshInstance == meshInstance.get())
            {
                //ImGui::PushStyleColor(imguicol_te, ImVec4(0.0f, 1.0f, 0.0f, 1.0f));
            }

            if (ImGui::MenuItem(meshInstance->m_Name.c_str(), 0))
                m_SelectedMeshInstance = meshInstance.get();
        }
        ImGui::ListBoxFooter();
    }

    if (m_SelectedMeshInstance != nullptr)
    {
        auto& transform = m_SelectedMeshInstance->m_Transform;
        TransformEditor(transform);
    }
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

