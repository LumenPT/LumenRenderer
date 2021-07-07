#pragma once
#include "Transform.h"
#include "Lumen/Renderer/ILumenResources.h"

#include <string>

namespace Lumen
{
    // Base class for volume instances
    class VolumeInstance
    {
    public:
    	VolumeInstance()
    		: m_Name("Unnamed Volume Instance") {}
        virtual void SetVolume(std::shared_ptr<Lumen::ILumenVolume> a_Volume)
        {
            m_VolumeRef = a_Volume;
        };

        std::shared_ptr<Lumen::ILumenVolume> GetVolume() const { return m_VolumeRef; }

        float GetDensity() const { return m_Density; }
        virtual void SetDensity(float a_Density) { m_Density = a_Density; }

        Transform m_Transform;
        std::string m_Name;
    protected:
        std::shared_ptr<Lumen::ILumenVolume> m_VolumeRef;
		float m_Density = 0.001f;	//Default value
    };
}
