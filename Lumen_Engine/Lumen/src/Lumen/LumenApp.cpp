#include "lmnpch.h"
#include "LumenApp.h"
#include "Layer.h"
#include "ImGui/ImGuiLayer.h"
#include "GLTaskSystem.h"

#include "ModelLoading/SceneManager.h"

#include "Lumen/Log.h"

#include <glad/glad.h>

#include <gltf.h>	//temp

#include "Input.h"


namespace Lumen
{
#define BIND_EVENT_FN(x) std::bind(&LumenApp::x, this, std::placeholders::_1)

	LumenApp* LumenApp::s_Instance = nullptr;
	
	LumenApp::LumenApp()
	{
		LMN_CORE_ASSERT(!s_Instance, "Lumen Application already exists");
		s_Instance = this;
		
		m_Window = std::unique_ptr<Window>(Window::Create());
		m_Window->SetEventCallback(BIND_EVENT_FN(OnEvent));	//research placeholders

		m_ImGuiLayer = new ImGuiLayer();
		PushOverlay(m_ImGuiLayer);

		m_SceneManager = std::make_unique<SceneManager>();

		// Fill out the service locator for the layers
		m_LayerServices.m_SceneManager = m_SceneManager.get();

		Input::SetCallbacks();
		GLTaskSystem::Initialize(reinterpret_cast<GLFWwindow*>(m_Window->GetNativeWindow()));
	}

	LumenApp::~LumenApp()
	{
		GLTaskSystem::Destroy();
		m_SceneManager.reset();
	}

	void LumenApp::Run()
	{
		while (m_Running)
		{
			//a hack, very dirty. This shouldn't be here
			glClearColor(1, 1, 0.5f, 1);
			glClear(GL_COLOR_BUFFER_BIT);


			//Update loop
			for (Layer* layer: m_LayerStack)
			{
				layer->OnUpdate();
				//LMN_INFO("{0}", layer->GetName());
			}

			//Render loop standin
			m_ImGuiLayer->Begin();
			for (Layer* layer : m_LayerStack)
			{
				layer->OnDraw();
				layer->OnImGuiRender();
			}
			m_ImGuiLayer->End();
			Input::Update();

			m_Window->OnUpdate();
		}
	}

	void LumenApp::OnEvent(Event& e)
	{
		EventDispatcher dispatcher(e);

		dispatcher.Dispatch<WindowCloseEvent>(BIND_EVENT_FN(OnWindowClosed));
		//LMN_CORE_TRACE("{0}", e);

		for (auto it = m_LayerStack.end(); it != m_LayerStack.begin(); )
		{
			(*--it)->OnEvent(e);
			if(e.m_Handled)
			{
				break;
			}
		}
	}

	void LumenApp::PushLayer(Layer* layer)
	{
		layer->SetLayerServices(&m_LayerServices);
		m_LayerStack.PushLayer(layer);
		layer->OnAttach();
	}

	void LumenApp::PushOverlay(Layer* layer)
	{
		m_LayerStack.PushOverlay(layer);
		layer->OnAttach();
	}

	bool LumenApp::OnWindowClosed(WindowCloseEvent &e)
	{
		m_Running = false;
		return true;
	}

}
