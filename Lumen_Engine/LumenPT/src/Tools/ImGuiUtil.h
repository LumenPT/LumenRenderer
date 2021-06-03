#pragma once

#include <glm/vec2.hpp>

#include "cstdint"
#include "Renderer/Camera.h"

class CudaGLTexture;

namespace ImGuiUtil
{
    using GLuint = uint32_t;
    void DisplayImage(GLuint a_GLTexture, glm::ivec2 a_Size, glm::vec2 a_UV1 = glm::vec2(0.0f), glm::vec2 a_UV2 = glm::vec2(1.0f));
    void DisplayImage(CudaGLTexture& a_CudaGLTexture, glm::ivec2 = glm::ivec2(0), glm::vec2 a_UV1 = glm::vec2(0.0f), glm::vec2 a_UV2 = glm::vec2(1.0f));
}