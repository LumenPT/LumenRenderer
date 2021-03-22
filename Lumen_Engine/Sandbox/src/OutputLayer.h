#pragma once

#include "Lumen/Layer.h"

#include <cstdint>

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

    void OnAttach() override { InitializeScenePresets(); };

    void OnUpdate() override;

    void OnImGuiRender() override;

    //LumenPT* GetPipeline() { return m_LumenPT.get(); };
    LumenRenderer* GetPipeline() { return m_Renderer.get(); };

private:

    void InitializeScenePresets();
    void HandleCameraInput(Camera& a_Camera);
    void HandleSceneInput();
    void ImGuiCameraSettings();
    std::unique_ptr<LumenRenderer> m_Renderer;
    //std::unique_ptr<LumenPT> m_LumenPT;

    uint32_t m_Program;
    
    std::vector<ScenePreset> m_ScenePresets;

    float m_CameraMouseSensitivity;
    float m_CameraMovementSpeed;

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
    "uniform sampler2D u_Texture;\n                                                           "
    "                                                                                         "
    "void main()\n                                                                            "
    "{                                                                                        "
    "    FragColor = texture(u_Texture, a_UV);                                                "
    "}                                                                                        ";
};