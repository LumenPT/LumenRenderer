#pragma once

class LumenRenderer;

namespace Lumen
{
    class ILumenScene;
    class MeshInstance;
    class VolumeInstance;
    class Transform;

    class SceneGraph
    {
    public:

        SceneGraph();
    	
        void SetRendererRef(LumenRenderer& a_Renderer) { m_RendererRef = &a_Renderer; }

        void Display(ILumenScene& a_Scene);

    private:
        void TransformEditor(Transform& a_Transform);

        LumenRenderer* m_RendererRef;

        MeshInstance* m_SelectedMeshInstance;
        VolumeInstance* m_SelectedVolumeInstance;
        std::string m_SearchString;
    };

}