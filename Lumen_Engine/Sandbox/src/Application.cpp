#include <Lumen.h>

#include "OutputLayer.h"

#include "imgui/imgui.h"

#include "GLFW/include/GLFW/glfw3.h"


//#include "imgui/imgui.h"

namespace Lumen {
    class WindowsWindow;
}

namespace LumenPTConsts
{
	const static std::string ShaderPath = "whatever";
}

class ExampleLayer : public Lumen::Layer
{
public:
	ExampleLayer()
		: Layer("Example")
	{
		
	}
	
	void OnUpdate() override
	{
		if(Lumen::Input::IsKeyPressed(LMN_KEY_TAB))
			LMN_INFO("Tab  key is pressed");
	}

	virtual void OnImGuiRender() override
	{
		ImGui::Begin("Test");
		ImGui::Text("Hello Lumen Renderer Engine, We gon' path trace the hell outta you.");
		ImGui::End();
	}

	void OnEvent(Lumen::Event& event) override
	{

		if(event.GetEventType() == Lumen::EventType::KeyPressed)
		{
			Lumen::KeyPressedEvent& e = static_cast<Lumen::KeyPressedEvent&>(event);
			LMN_TRACE("{0}", static_cast<char>(e.GetKeyCode()));
		}
	}
};


class Sandbox : public Lumen::LumenApp
{
public:
	Sandbox()
	{
		glfwMakeContextCurrent(reinterpret_cast<GLFWwindow*>(GetWindow().GetNativeWindow()));

		PushLayer(new ExampleLayer());
		PushLayer(new OutputLayer());

		//PushOverlay(new Lumen::ImGuiLayer());
		
		//Lumen::SceneManager Manager;
		//Manager.LoadGLTF("Yes lets load");
	}

	~Sandbox()
	{
		
	}
};

Lumen::LumenApp* Lumen::CreateApplication()
{
	return new Sandbox();
}
