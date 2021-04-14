#pragma once
#include "Transform.h"
#include "Lumen/Renderer/ILumenResources.h"

namespace Lumen
{
    // Base class for volume instances
    class VolumeInstance
    {
    public:
        virtual void SetVolume(std::shared_ptr<Lumen::ILumenVolume> a_Volume)
        {
            m_VolumeRef = a_Volume;
        };

        std::shared_ptr<Lumen::ILumenVolume> GetVolume() const { return m_VolumeRef; }

        Transform m_Transform;
    protected:
        std::shared_ptr<Lumen::ILumenVolume> m_VolumeRef;
    };
}
