#pragma once
#include <glm/vec3.hpp>
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

#include <memory>
#include <vector>

namespace Lumen
{
	class VertexBuffer
	{
	public:
		virtual ~VertexBuffer() = 0;
	};

	class LumenRenderer
	{
	public:
		LumenRenderer();

		virtual std::unique_ptr<VertexBuffer> CreateVertexBuffer() = 0;
		virtual std::unique_ptr<VertexBuffer>CreateIndexBuffer() = 0;
		void CreateTexture();
		
	private:
		

		
};
}