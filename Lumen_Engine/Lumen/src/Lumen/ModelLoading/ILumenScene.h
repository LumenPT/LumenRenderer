#pragma once
#include "MeshInstance.h"
#include "VolumeInstance.h"
#include "Lumen/Renderer/Camera.h"

namespace Lumen
{
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

        virtual Lumen::MeshInstance* AddMesh();
        virtual Lumen::VolumeInstance* AddVolume();

        virtual void Clear();

        std::vector<std::unique_ptr<Lumen::MeshInstance>> m_MeshInstances;
        std::vector<std::unique_ptr<Lumen::VolumeInstance>> m_VolumeInstances;
        const std::unique_ptr<Camera> m_Camera;
    	//accelleration structure

    private:
    };
}
