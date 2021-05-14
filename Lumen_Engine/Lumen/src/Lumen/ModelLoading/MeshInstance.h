#pragma once
#include <glm/glm.hpp>

#include "Transform.h"
#include "Lumen/Renderer/ILumenResources.h"

namespace Lumen
{
    /*
     * Enumeration to set how a mesh emission should behave.
     */
    enum class EmissionMode
    {  
        ENABLED,    //Emissive triangles emit light.
        DISABLED,   //No triangles emit light.
        OVERRIDE    //All triangles emit light, despite what the triangle base values are.
    };

    // Base class for mesh instances
    class MeshInstance
    {
    public:
        virtual ~MeshInstance() = default;

        virtual void SetMesh(std::shared_ptr<Lumen::ILumenMesh> a_Mesh)
        {
            m_MeshRef = a_Mesh;
        };

        //TODO enable this
        /*
         * Set the emission values for this mesh.
         * The mode determines how emission works.
         * The radiance is the RGB radiance value when mode is set to OVERRIDE.
         * Scale scales up the radiance value.
         */
        //virtual void SetEmissiveness(const EmissionMode a_Mode, const glm::vec3& a_OverrideRadiance, const float a_Scale) = 0;

        //TODO override material for all primitives in mesh.

        auto GetMesh() const { return m_MeshRef; }

        Transform m_Transform;
    protected:
        std::shared_ptr<Lumen::ILumenMesh> m_MeshRef;
    };
}
