#pragma once
#include "Core.h"
#include "Window.h"
#include "Lumen/LayerStack.h"
#include "Lumen/Events/Event.h"
#include "Lumen/Events/ApplicationEvent.h"
#include "Lumen/Renderer/LumenRenderer.h"

//#include "Lumen/ImGui/ImGuiLayer.h"

namespace Lumen
{
	class SceneManager;
	class ImGuiLayer;
	class LumenApp
	{
	public:

		// Struct used to provide application layers with specific services
		struct LayerServices
		{
			SceneManager* m_SceneManager;
		};

		LumenApp();
		virtual ~LumenApp();

		void Run();

		void OnEvent(Event& e);

		void PushLayer(Layer* layer);
		void PushOverlay(Layer* layer);

		inline Window& GetWindow() { return *m_Window; }
		inline static LumenApp& Get() { return *s_Instance; }

	protected:
		std::unique_ptr<SceneManager> m_SceneManager;

	private:
		bool OnWindowClosed(WindowCloseEvent& e);
		
		std::unique_ptr<Window> m_Window;
		ImGuiLayer* m_ImGuiLayer;
		bool m_Running = true;
		LayerStack m_LayerStack;

		LayerServices m_LayerServices;


		static LumenApp* s_Instance;
	};

	//To be defined in CLIENT
	LumenApp* CreateApplication();
}

