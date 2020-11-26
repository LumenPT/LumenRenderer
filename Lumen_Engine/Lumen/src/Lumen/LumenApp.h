#pragma once
#include "Core.h"
#include "Window.h"
#include "Lumen/LayerStack.h"
#include "Lumen/Events/Event.h"
#include "Lumen/Events/ApplicationEvent.h"

#include "Lumen/ImGui/ImGuiLayer.h"

namespace Lumen
{

	class LumenApp
	{
	public:
		LumenApp();
		virtual ~LumenApp();

		void Run();

		void OnEvent(Event& e);

		void PushLayer(Layer* layer);
		void PushOverlay(Layer* layer);

		inline Window& GetWindow() { return *m_Window; }
		inline static LumenApp& Get() { return *s_Instance; }
		
	private:
		bool OnWindowClosed(WindowCloseEvent& e);
		
		std::unique_ptr<Window> m_Window;
		ImGuiLayer* m_ImGuiLayer;
		bool m_Running = true;
		LayerStack m_LayerStack;

		static LumenApp* s_Instance;
	};

	//To be defined in CLIENT
	LumenApp* CreateApplication();
}

