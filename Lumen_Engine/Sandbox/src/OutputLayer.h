#pragma once

#include "Lumen/Layer.h"

#include <cstdint>



#include "Renderer/Camera.h"
#include "Renderer/Camera.h"
#include "Tools/FrameSnapshot.h"

#include <deque>

namespace Lumen
{
    class SceneGraph;
}

class Camera;

class LumenRenderer;



class OutputLayer : public Lumen::Layer
{
    struct ScenePreset
    {
        uint16_t m_Key;
        std::function<void()> m_Function;
    };

public:
    OutputLayer();
    ~OutputLayer();

    void OnAttach() override;

    void OnUpdate() override;

    void OnImGuiRender() override;

    void OnEvent(Lumen::Event& a_Event) override;

    //LumenPT* GetPipeline() { return m_LumenPT.get(); };
    void SetPipeline(const std::shared_ptr<LumenRenderer>& a_Renderer);
    const std::shared_ptr<LumenRenderer>& GetPipeline() const;

private:

    void InitializeScenePresets();
    void HandleCameraInput(Camera& a_Camera);
    void HandleSceneInput();
    void ImGuiCameraSettings();
    void ImGuiPixelDebugger();

    void InitContentViewNameTable();
    void ContentViewDropDown();

    void MakeScreenshot(std::string a_ScreenshotFileName);
    std::string DefaultScreenshotName();

    std::shared_ptr<LumenRenderer> m_Renderer;
    //std::unique_ptr<LumenPT> m_LumenPT;

    uint32_t m_Program;
    
    std::vector<ScenePreset> m_ScenePresets;

    float m_CameraMouseSensitivity;
    float m_CameraMovementSpeed;
    float m_Gamma = 2.2f;
    float m_MinMaxRenderDistance[2] = { 0.1f, 1000.f };

    std::unique_ptr<class ModelLoaderWidget> m_ModelLoaderWidget;
    std::unique_ptr<Lumen::SceneGraph> m_SceneGraph;
    std::unique_ptr<class Profiler> m_Profiler;

    enum ContentViewMode
    {
        NONE = 0,
        BYTE,
        INT,
        FLOAT,
        INT2,
        FLOAT2,
        BYTE3,
        INT3,
        FLOAT3,
        BYTE4,
        INT4,
        FLOAT4,
        CONTENTVIEWMODE_COUNT
    };

    struct
    {
        bool m_SceneGraph = true;
        bool m_FileLoader = false;
        bool m_ImageResizer = false;
        bool m_PixelDebugger = false;
        bool m_CameraSettings = false;
        bool m_DebugViewport = false;
        bool m_Profiler = false;
        bool m_GeneralSettings = true;
    } m_EnabledTools;

    struct CmpNoChange
    {
        bool operator()(const std::string& a_A, const std::string& a_B) const
        {
            auto as = a_A.substr(0, a_A.find('p'));
            auto bs = a_B.substr(0, a_B.find('p'));

            auto a = std::stoi(as);
            auto b = std::stoi(bs);

            return a < b;
        }
    };

    inline static std::map<std::string, glm::uvec2, CmpNoChange> ms_PresetSizes = {
        {"480p", glm::uvec2(854, 480)},
        {"720p", glm::uvec2(1280, 720)},
        {"1080p", glm::uvec2(1920, 1080)},
        {"1440p", glm::uvec2(2560, 1440)},
        {"2160p", glm::uvec2(3840, 2160)}
    };

    int m_Dlss_SelectedMode = 2;    // 2 translates to "BALANCED" dlss mode 
    bool m_BlendMode = true;

    std::vector<std::unique_ptr<FrameSnapshot>> m_FrameSnapshots;
    int m_CurrentSnapShotIndex;
    const std::pair<const std::string, FrameSnapshot::ImageBuffer>* m_CurrentImageBuffer;

    glm::vec2 m_SnapshotUV1;
    glm::vec2 m_SnapshotUV2;

    ContentViewMode m_CurrContentView;
    std::function<void(glm::vec2)> m_ContentViewFunc;
    std::map<ContentViewMode, std::string> m_ContentViewNames;

    std::deque<FrameStats> m_PreviousFramesStats;
    const uint32_t m_MaxStoredFrames = 5 * 60 * 60; // 5 minutes of running at 60FPS
    uint32_t m_BarsDisplayed;

    uint32_t m_LastFrameTex;
    uint32_t m_SmallViewportFrameTex;

    inline static const char* m_VSSource = "#version 330 core \n                                                                  "
    "                                                                                                                             "
    "out vec2 a_UV; // UV coordinates    \n                                                                                       "
    "                                                                                                                             "
    "const vec2 gs_Positions[3] = vec2[3](                                                                                        "
    "   vec2(3.0,-1.0),                                                                                                           "
    "   vec2(-1.0, 3.0),                                                                                                          "
    "   vec2(-1.0,-1.0)                                                                                                           "
    ");                                                                                                                           "
    "                                                                                                                             "
    "const vec2 gs_TexCoords[3] = vec2[3](                                                                                        "
    "    vec2(2.0,0.0),                                                                                                           "
    "    vec2(0.0,-2.0),                                                                                                          "
    "    vec2(0.0,0.0)                                                                                                            "
    ");                                                                                                                           "
    "void main()\n                                                                                                                "
    "{                                                                                                                            "
    "    gl_Position = vec4(gs_Positions[gl_VertexID], 0.0, 1.0); // see how we directly give a vec3 to vec4's constructor\n      "
    "    a_UV = gs_TexCoords[gl_VertexID];       // set the output variable to a dark-red color\n                                 "
    "}                                                                                                                            ";

    inline static const char* m_FSSource =
    "#version 330 core\n                                                                      "
    "out vec4 FragColor;\n                                                                    "
    "                                                                                         "
    "in vec2 a_UV; // the input variable from the vertex shader (same name and same type)\n   "
    "                                                                                         "
    "uniform float a_Gamma; // gamma correction strength \n                                   "
    "uniform sampler2D u_Texture;\n                                                           "
    "                                                                                         "
    "void main()\n                                                                            "
    "{                                                                                        "
    "    FragColor = texture(u_Texture, a_UV);                                                "
    "    FragColor.rgb = pow(FragColor.rgb, vec3(1.0/a_Gamma));                               "
    "}                                                                                        ";
};