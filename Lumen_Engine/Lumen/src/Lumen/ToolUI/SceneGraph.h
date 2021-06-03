#pragma once

namespace Lumen
{
    class ILumenScene;
    class MeshInstance;
    class Transform;

    class SceneGraph
    {
    public:

        void Display(ILumenScene& a_Scene);

    private:
        void TransformEditor(Transform& a_Transform);

        MeshInstance* m_SelectedMeshInstance;
    };

}