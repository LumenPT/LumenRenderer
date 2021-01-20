#pragma once
#include "MeshInstance.h"
#include "VolumeInstance.h"

namespace Lumen
{
    class ILumenScene
    {
    public:
        ILumenScene() {};
        virtual ~ILumenScene() {};

        virtual Lumen::MeshInstance* AddMesh();
        virtual Lumen::VolumeInstance* AddVolume();

        std::vector<std::unique_ptr<Lumen::MeshInstance>> m_MeshInstances;
        std::vector<std::unique_ptr<Lumen::VolumeInstance>> m_VolumeInstances;

    private:
    };
}
