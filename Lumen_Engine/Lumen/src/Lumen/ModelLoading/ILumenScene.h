#pragma once
#include "MeshInstance.h"
#include "VolumeInstance.h"
#include "Lumen/Renderer/Camera.h"

#include <string>

namespace Lumen
{
    // Interface class for scenes
    class ILumenScene
    {
    public:
        struct Node
        {
            Node()
                : m_MeshInstancePtr(nullptr)
                , m_VolumeInstancePtr(nullptr)
                , m_Parent(nullptr)
                , m_Name("Unnamed Node")
                , m_ScenePtr(nullptr)
            {}

            Node* AddChild();

            void AddChild(std::unique_ptr<Lumen::ILumenScene::Node>& a_Node);
            void RemoveChild(std::unique_ptr<Lumen::ILumenScene::Node>& a_Node);

            Node* GetFirstIntermediateNode(const Node* a_ParentNode) const;

            bool IsChildOf(const Node& a_Node) const;

            Transform m_Transform; 
            std::string m_Name;
            Node* m_Parent;
            ILumenScene* m_ScenePtr; // Initialized to the scene pointer for the root node 
            std::vector<std::unique_ptr<Node>> m_ChildNodes;
            Lumen::MeshInstance* m_MeshInstancePtr;
            Lumen::VolumeInstance* m_VolumeInstancePtr;
        };

        /// <summary>
        /// Takes in camera data on initialization
        /// </summary>
        /// <param name="a_CamPosition"></param>
        /// <param name="a_CamUp"></param>
        ILumenScene(glm::vec3 a_CamPosition = glm::vec3(0, 0, -50.f), glm::vec3 a_CamUp = glm::vec3(0, 1, 0))
    	: m_Camera(std::make_unique<Camera>(a_CamPosition, a_CamUp)) {};
        virtual ~ILumenScene() {};

        // Adds a mesh instance to the scene. Use this instead of manually adding instances to m_MeshInstances
        // in order to ensure the instances are of the correct type.
        virtual Lumen::MeshInstance* AddMesh();
        // Adds a volume instance to the scene. Use this instead of manually adding instances to m_VolumeInstances
        // in order to ensure the instances are of the correct type.
        virtual Lumen::VolumeInstance* AddVolume();

        // Removes all instances from the scene
        virtual void Clear();

        std::string m_Name;
        std::vector<std::unique_ptr<Lumen::MeshInstance>> m_MeshInstances;
        std::vector<unsigned int> m_MeshLightIndices;
        std::vector<std::unique_ptr<Lumen::VolumeInstance>> m_VolumeInstances;
        const std::unique_ptr<Camera> m_Camera;
        std::vector<std::unique_ptr<Node>> m_RootNodes; // GLTF allows for multiple root nodes in the same scene, so support that
    	//accelleration structure

    private:
    };
}
