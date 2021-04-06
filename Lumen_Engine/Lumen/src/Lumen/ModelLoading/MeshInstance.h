#pragma once
#include "Transform.h"
#include "Lumen/Renderer/ILumenResources.h"

namespace Lumen
{
    // Base class for mesh instances
    class MeshInstance
    {
    public:
        virtual void SetMesh(std::shared_ptr<Lumen::ILumenMesh> a_Mesh)
        {
            m_MeshRef = a_Mesh;
        };

        auto GetMesh() const { return m_MeshRef; }

        Transform m_Transform;
    protected:
        std::shared_ptr<Lumen::ILumenMesh> m_MeshRef;
    };
}
