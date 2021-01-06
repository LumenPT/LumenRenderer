#pragma once
#include <glm/fwd.hpp>
#include <memory>

namespace Lumen
{
	class ILumenTexture
	{
	public:
		virtual ~ILumenTexture(){};
	};

    class ILumenMaterial
    {
    public:
        virtual void SetDiffuseColor(const glm::vec4& a_NewDiffuseColor) = 0;
        virtual void SetDiffuseTexture(std::shared_ptr<ILumenTexture> a_NewDiffuseTexture) = 0;

        virtual glm::vec4 GetDiffuseColor() const = 0;
        virtual ILumenTexture& GetDiffuseTexture() const = 0;
    };
}
