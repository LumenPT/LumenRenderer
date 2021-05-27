#pragma once
#include "Transform.h"
#include "Lumen/Renderer/ILumenResources.h"

namespace Lumen
{
    // Base class for mesh instances
    class MeshInstance
    {
    public:
        MeshInstance()
        : m_AdditionalColor(1.0f, 1.0f, 1.0f, 1.0f)
        {
            
        }

        virtual void SetMesh(std::shared_ptr<Lumen::ILumenMesh> a_Mesh)
        {
            m_MeshRef = a_Mesh;
        };

        auto GetMesh() const { return m_MeshRef; }


        virtual void SetAdditionalColor(glm::vec4 a_AdditionalColor) { m_AdditionalColor = a_AdditionalColor; };


        Transform m_Transform;
    protected:
        std::shared_ptr<Lumen::ILumenMesh> m_MeshRef;
        glm::vec4 m_AdditionalColor;
    };
}
