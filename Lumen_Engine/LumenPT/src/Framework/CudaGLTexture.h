#pragma once

#include <glm/vec2.hpp>

#include "Cuda/cuda_runtime.h"
#include "Glad/glad.h"

#include "cstdint"

class CudaGLTexture
{
public:
    CudaGLTexture(GLuint a_Format = GL_RGBA8, uint32_t a_Width = 0, uint32_t a_Height = 0, uint8_t a_PixelSize = 4);
    ~CudaGLTexture();

    template<typename T = uchar4>
    T* GetDevicePtr() {
        size_t size;
        void* ptr;
        cudaGraphicsResourceGetMappedPointer(&ptr, &size, m_CudaGraphicsResource);
        return reinterpret_cast<T*>(ptr);
    };

    GLuint GetTexture();

    void Resize(uint32_t a_Width, uint32_t a_Height);

    glm::ivec2 GetSize() const { return glm::ivec2(m_Width, m_Height); }

private:

    void UpdateTexture();

    uint32_t m_Width;
    uint32_t m_Height;
    uint8_t m_PixelSize;  

    GLuint m_PixelBuffer;
    GLuint m_Texture;
    GLuint m_Format;

    cudaGraphicsResource* m_CudaGraphicsResource;

};