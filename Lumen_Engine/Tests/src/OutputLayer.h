#pragma once
#include <Lumen/Layer.h>

class LumenRenderer;

class OutputLayer : public Lumen::Layer
{

public:

    OutputLayer();
    ~OutputLayer();

    //void OnUpdate() override;

    void OnDraw() override;

    void SetPipeline(std::unique_ptr<LumenRenderer>& a_Pipeline);
    LumenRenderer& GetPipeline() const;

private:

    bool Init();

    std::unique_ptr<LumenRenderer> m_Renderer;
    GLuint m_OGLProgramId;

    inline static const char* m_VSSource = 
        "#version 330 core \n                                                                                                         "
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
