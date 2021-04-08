#include "Application.h"
#include <Lumen/ModelLoading/SceneManager.h>

#ifdef WAVEFRONT
#include <Framework/WaveFrontRenderer.h>
#elif
#include <Framework/OptixRenderer.h>
#endif

#include "GLFW/include/GLFW/glfw3.h"
#include <filesystem>

Lumen::LumenApp* Lumen::CreateApplication()
{

    return new Tests();

}



Tests::Tests()
{

    Init();

}

Tests::~Tests()
{}

bool Tests::Init()
{

    //Set glfw context
    glfwMakeContextCurrent(reinterpret_cast<GLFWwindow*>(this->GetWindow().GetNativeWindow()));

    {
#ifdef WAVEFRONT

        std::unique_ptr<LumenRenderer> renderer = std::make_unique<WaveFront::WaveFrontRenderer>();

        //TODO get settings from file or command line to test a configuration.
        WaveFront::WaveFrontSettings settings{};
        settings.depth = 3;
        settings.minIntersectionT = 0.1f;
        settings.maxIntersectionT = 5000.f;
        settings.renderResolution = { 800, 600 };
        settings.outputResolution = { 800, 600 };

        static_cast<WaveFront::WaveFrontRenderer*>(renderer.get())->Init(settings);

#elif

        LumenRenderer::InitializationData init{};
        init.m_RenderResolution = { 800, 600 };
        init.m_OutputResolution = { 800, 600 };
        init.m_MaxDepth = 1;
        init.m_RaysPerPixel = 1;
        init.m_ShadowRaysPerPixel = 1;

        auto renderer = std::make_unique<OptixRenderer>(init);

#endif

        m_OutputLayer = std::make_unique<OutputLayer>();
        m_OutputLayer->SetPipeline(renderer);
        PushLayer(m_OutputLayer.get());
    }

    const std::filesystem::path workingDir = std::filesystem::current_path();
    std::string workingDirPath = workingDir.string();
    std::replace(workingDirPath.begin(), workingDirPath.end(), '\\', '/');

    const std::string meshPath = workingDirPath.append("/Sandbox/assets/models/Sponza/");
    const std::string meshFile = "Sponza.gltf";

    auto& renderer = m_OutputLayer->GetPipeline();

    m_SceneManager->SetPipeline(renderer);

    auto resource = m_SceneManager->LoadGLTF(meshFile, meshPath);

    const LumenRenderer::SceneData sceneData{};
    renderer.m_Scene = renderer.CreateScene(sceneData);

    auto mesh = renderer.m_Scene->AddMesh();
    mesh->SetMesh(resource->m_MeshPool[0]);

    return true;

}
