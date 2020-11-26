#pragma once

#include "Lumen/Layer.h"

#include "Lumen/Events/ApplicationEvent.h"
#include "Lumen/Events/KeyEvent.h"
#include "Lumen/Events/MouseEvent.h"

namespace Lumen
{
	class ImGuiLayer : public Layer
	{
	public:
		ImGuiLayer();
		~ImGuiLayer();

		virtual void OnAttach() override;
		virtual void OnDetach() override;
		virtual void OnImGuiRender() override;

		void Begin();
		void End();
	private:
		
		float m_Time = 0.0f;
		
	};


	
}


