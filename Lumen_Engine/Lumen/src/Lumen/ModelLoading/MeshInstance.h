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
        struct Emissiveness
        {
            Emissiveness(EmissionMode a_EmissionMode = EmissionMode::ENABLED, glm::vec3 a_Emission = glm::vec3(0.0f), float a_EmissionScale = 1.0f)
                : m_EmissionMode(a_EmissionMode)
                , m_OverrideRadiance(a_Emission)
                , m_Scale(a_EmissionScale) {}
            EmissionMode m_EmissionMode;
            glm::vec3 m_OverrideRadiance;
            float m_Scale;
        };

        virtual ~MeshInstance() = default;

        MeshInstance()
            : m_AdditionalColor(1.0f, 1.0f, 1.0f, 1.0f)
            , m_EmissiveProperties()
    		, m_PreviousEmissionMode(m_EmissiveProperties.m_EmissionMode)
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
        virtual void SetEmissiveness(const Emissiveness& a_EmissiveProperties)
        {
            auto em = m_EmissiveProperties.m_EmissionMode;
            m_EmissiveProperties = a_EmissiveProperties;
            if (m_EmissiveProperties.m_EmissionMode != EmissionMode::OVERRIDE)
                m_EmissiveProperties.m_EmissionMode = m_PreviousEmissionMode;
            else if (em != m_EmissiveProperties.m_EmissionMode)
                m_PreviousEmissionMode = em;
        }

        const Emissiveness& GetEmissiveness() const { return m_EmissiveProperties; };

        virtual void UpdateAccelRemoveThis() {}

        /*
         * Get this instances emission mode.
         */
        virtual EmissionMode GetEmissionMode()
        {
            return m_EmissiveProperties.m_EmissionMode;
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

        std::shared_ptr<Lumen::ILumenMaterial> GetOverrideMaterial()
        {
            return m_OverrideMaterial;
        }

        auto GetMesh() const { return m_MeshRef; }


        virtual void SetAdditionalColor(glm::vec4 a_AdditionalColor) { m_AdditionalColor = a_AdditionalColor; };


        Transform m_Transform;
        std::string m_Name;
    protected:
        //Emissive instance properties.
        Emissiveness m_EmissiveProperties;
        EmissionMode m_PreviousEmissionMode;

        //Material to override with.
        std::shared_ptr<Lumen::ILumenMaterial> m_OverrideMaterial;

        std::shared_ptr<Lumen::ILumenMesh> m_MeshRef;
        glm::vec4 m_AdditionalColor;
    };
}
