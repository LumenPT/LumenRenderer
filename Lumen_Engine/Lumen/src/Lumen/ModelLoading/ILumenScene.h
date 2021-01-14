#pragma once
#include "MeshInstance.h"

namespace Lumen
{
    class ILumenScene
    {
    public:
        ILumenScene() {};
        virtual ~ILumenScene(){};

        virtual Lumen::MeshInstance* AddMesh();

        std::vector<std::unique_ptr<Lumen::MeshInstance>> m_MeshInstances;
    private:
    };
}
