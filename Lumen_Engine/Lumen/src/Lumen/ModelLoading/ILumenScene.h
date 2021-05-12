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
    	//accelleration structure

    private:
    };
}
