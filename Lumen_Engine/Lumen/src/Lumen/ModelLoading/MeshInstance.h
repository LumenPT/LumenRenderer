#pragma once
#include <glm/glm.hpp>

#include "Transform.h"
#include "Lumen/Renderer/ILumenResources.h"

#include <string>

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

        MeshInstance()
            : m_AdditionalColor(1.0f, 1.0f, 1.0f, 1.0f)
            , m_EmissiveOverride(1.f)
            , m_EmissionScale(1.f)
            , m_EmissionMode(Lumen::EmissionMode::ENABLED)
            , m_Name("Unnamed mesh instance")
        {
            
        }

        virtual void SetMesh(std::shared_ptr<Lumen::ILumenMesh> a_Mesh)
        {
            m_MeshRef = a_Mesh;
        };

        /*
         * Set the emission values for this mesh.
         * The mode determines how emission works.
         * The radiance is the RGB radiance value when mode is set to OVERRIDE.
         * Scale scales up the radiance value.
         */
        virtual void SetEmissiveness(const EmissionMode a_Mode, const glm::vec3& a_OverrideRadiance, const float a_Scale)
        {
            m_EmissionMode = a_Mode;
            m_EmissiveOverride = a_OverrideRadiance;
            m_EmissionScale = a_Scale;
        }

        virtual void UpdateAccelRemoveThis() {}

        /*
         * Get this instances emission mode.
         */
        virtual EmissionMode GetEmissionMode()
        {
            return m_EmissionMode;
        }

        /*
         * Override the material used by this mesh.
         * Has to be non-null material.
         */
        virtual void SetOverrideMaterial(std::shared_ptr<Lumen::ILumenMaterial> a_OverrideMaterial)
        {
            assert(a_OverrideMaterial != nullptr);
            m_OverrideMaterial = a_OverrideMaterial;
        }

        auto GetMesh() const { return m_MeshRef; }


        virtual void SetAdditionalColor(glm::vec4 a_AdditionalColor) { m_AdditionalColor = a_AdditionalColor; };


        Transform m_Transform;
        std::string m_Name;
    protected:
        //Emissive instance properties.
        glm::vec3 m_EmissiveOverride;       //Emission color if overridden.
        Lumen::EmissionMode m_EmissionMode; //Emission mode.
        float m_EmissionScale;              //Emission scale.

        //Material to override with.
        std::shared_ptr<Lumen::ILumenMaterial> m_OverrideMaterial;

        std::shared_ptr<Lumen::ILumenMesh> m_MeshRef;
        glm::vec4 m_AdditionalColor;
    };
}
