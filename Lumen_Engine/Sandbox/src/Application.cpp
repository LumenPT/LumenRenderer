#include <Lumen.h>
#include <string>
#include <filesystem>
#include <algorithm>

#include "OutputLayer.h"

#include "imgui/imgui.h"
#include "ModelLoading/Node.h"

#include "GLFW/include/GLFW/glfw3.h"
#include "Lumen/ModelLoading/SceneManager.h"

#include "AppConfiguration.h"
#include "Framework/CudaUtilities.h"

#include <chrono>

#ifdef WAVEFRONT
#include "../../LumenPT/src/Framework/WaveFrontRenderer.h"
#else
#include "../../LumenPT/src/Framework/OptiXRenderer.h"
#endif


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
			//LMN_TRACE("{0}", static_cast<char>(e.GetKeyCode()));
		}
	}
};


class Sandbox : public Lumen::LumenApp
{
public:
	Sandbox()
	{
		glfwMakeContextCurrent(reinterpret_cast<GLFWwindow*>(GetWindow().GetNativeWindow()));
		//PushOverlay(new Lumen::ImGuiLayer());

		const std::filesystem::path configFilePath = std::filesystem::current_path() += "/Config.json";

		AppConfiguration& config = AppConfiguration::GetInstance();
		config.Load(configFilePath, true, true);

		std::shared_ptr<LumenRenderer> renderer = nullptr;

#ifdef WAVEFRONT

		renderer = std::make_shared<WaveFront::WaveFrontRenderer>();
		WaveFront::WaveFrontSettings settings{};

		settings.m_ShadersFilePathSolids = config.GetDirectoryShaders().string() +  config.GetFileShaderSolids().string();
		settings.m_ShadersFilePathVolumetrics = config.GetDirectoryShaders().string() + config.GetFileShaderVolumetrics().string();

		settings.depth = 5;
		settings.renderResolution = uint2{ 1280, 720 };
		//settings.outputResolution = { 2560, 1440 };
		settings.outputResolution = uint2{ 1280, 720 };
		settings.blendOutput = false;	//When true will blend output instead of overwriting it (high res image over time if static scene).

		std::static_pointer_cast<WaveFront::WaveFrontRenderer>(renderer)->Init(settings);

		CHECKLASTCUDAERROR;
		renderer->CreateDefaultResources();


#else

		OptiXRenderer::InitializationData initData;
		initData.m_AssetDirectory = config.GetDirectoryAssets();
		initData.m_ShaderDirectory = config.GetDirectoryShaders();

		initData.m_MaxDepth = 5;
		initData.m_RaysPerPixel = 1;
		initData.m_ShadowRaysPerPixel = 1;

		initData.m_RenderResolution = { 800, 600 };
		initData.m_OutputResolution = { 800, 600 };

		renderer = std::make_unique<OptiXRenderer>(initData);
#endif

		OutputLayer* contextLayer = new OutputLayer;
		contextLayer->SetPipeline(renderer);
		PushLayer(contextLayer);

		if (config.HasDefaultModel())
		{

			const std::filesystem::path& modelDirectory = config.GetDirectoryModels();
			const std::filesystem::path& defaultModel = config.GetDefaultModel();
			

			LMN_TRACE(modelDirectory.string() + defaultModel.string());

			m_SceneManager->SetPipeline(*contextLayer->GetPipeline());

			auto begin = std::chrono::high_resolution_clock::now();

			auto res = m_SceneManager->LoadGLTF(defaultModel.string(), modelDirectory.string());

			auto end = std::chrono::high_resolution_clock::now();

			auto milli = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

			printf("\n\nTime elapsed to load model: %lli milliseconds\n\n", milli);

			auto lumenPT = contextLayer->GetPipeline();

			lumenPT->m_Scene = res->m_Scenes[0];
			lumenPT->m_Scene->m_Camera->SetPosition(glm::vec3{ -150.f, 300.f, 150.f });
			lumenPT->m_Scene->m_Camera->SetRotation(glm::quatLookAtRH(glm::normalize(glm::vec3{ -1.f, 0.5f, 1.f }), glm::vec3{ 0.f, 1.f, 0.f }));

		}

		//renderer->InitNGX();

		contextLayer->GetPipeline()->StartRendering();

	}

	~Sandbox()
	{
		
	}
};

Lumen::LumenApp* Lumen::CreateApplication()
{
	return new Sandbox();
}
