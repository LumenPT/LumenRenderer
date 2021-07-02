#pragma once
#include <map>

#include "Lumen/ModelLoading/ILumenScene.h"

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

        void NodeSelection(Lumen::ILumenScene& a_Scene);
        void InstanceSelection(Lumen::ILumenScene& a_Scene);

        // Function called recursively to display nodes via Dear ImGui
        void DisplayNode(Lumen::ILumenScene::Node& a_Node, std::map<std::string, uint32_t>& a_NameMap, uint16_t a_Depth = 0);

        LumenRenderer* m_RendererRef;

        bool m_DisplayNodes;

        MeshInstance* m_SelectedMeshInstance;
        VolumeInstance* m_SelectedVolumeInstance;
        Lumen::ILumenScene::Node* m_SelectedNode;
        std::string m_SearchString;


        const glm::uvec2 m_GraphSize = glm::uvec2(300, 350);
    };

}