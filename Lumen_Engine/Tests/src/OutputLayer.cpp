#include "OutputLayer.h"


OutputLayer::OutputLayer()
{

    Init();

}

OutputLayer::~OutputLayer()
{}



inline bool CheckShaderCompilation(
    GLuint a_ShaderId, 
    const std::string& a_ShaderName = "")
{

    int success;
    char infoLog[512];
    glGetShaderiv(a_ShaderId, GL_COMPILE_STATUS, &success);

    if(!success)
    {
        glGetShaderInfoLog(a_ShaderId, sizeof(infoLog), nullptr, infoLog);
        std::cout << "ERROR: shader compilation failed: \n\t"
            << "Shader name: " << a_ShaderName.c_str() << "\n\t"
            << infoLog << std::endl;

    }

    return success;

}

inline bool CheckProgramLinking(GLuint a_ProgramId)
{
    int success;
    char infoLog[512];

    glGetProgramiv(a_ProgramId, GL_LINK_STATUS, &success);

    if(!success)
    {

        glGetProgramInfoLog(a_ProgramId, 512, nullptr, infoLog);
        std::cout << "ERROR: program linking failed: \n\t"
            << infoLog << std::endl;

    }

    return success;

}

bool OutputLayer::Init()
{

    const GLuint vsId = glCreateShader(GL_VERTEX_SHADER);
    const GLuint fsId = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(vsId, 1, &m_VSSource, nullptr);
    glCompileShader(vsId);

    if (!CheckShaderCompilation(vsId, "Vertex Shader"))
    {
        glDeleteShader(vsId);
        return false;
    }

    glShaderSource(fsId, 1, &m_FSSource, nullptr);
    glCompileShader(fsId);

    if (!CheckShaderCompilation(fsId, "Fragment Shader"))
    {
        glDeleteShader(fsId);
        return false;
    }

    GLuint programId = glCreateProgram();

    glAttachShader(programId, vsId);
    glAttachShader(programId, fsId);

    glLinkProgram(programId);

    if (!CheckProgramLinking(programId))
    {
        glDeleteProgram(programId);
        return false;
    }

    glDeleteShader(vsId);
    glDeleteShader(fsId);

    m_OGLProgramId = programId;

    return true;

}






void OutputLayer::OnDraw()
{

    if (!m_Renderer) { return; }
    const unsigned int textureId = m_Renderer->TraceFrame(m_Renderer->m_Scene);
    
    glBindTexture(GL_TEXTURE_2D, textureId);
    glUseProgram(m_OGLProgramId);
    glDrawArrays(GL_TRIANGLES, 0, 3);

}

void OutputLayer::SetPipeline(std::unique_ptr<class LumenRenderer>& a_Pipeline)
{

    m_Renderer = std::move(a_Pipeline);

}


LumenRenderer& OutputLayer::GetPipeline() const
{

    return *m_Renderer;

}
