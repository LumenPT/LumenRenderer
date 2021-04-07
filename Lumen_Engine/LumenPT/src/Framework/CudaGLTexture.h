#pragma once

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>


#include "Cuda/cuda_runtime.h"
#include "Glad/glad.h"

#include "cstdint"
#include <algorithm>

class CudaGLTexture
{
public:
    CudaGLTexture(GLuint a_Format = GL_RGBA8, uint32_t a_Width = 0, uint32_t a_Height = 0, uint8_t a_PixelSize = 4);
    ~CudaGLTexture();

    template<typename T = uchar4>
    T* GetDevicePtr() {
        m_TextureDirty = true;
        size_t size;
        void* ptr;
        cudaGraphicsResourceGetMappedPointer(&ptr, &size, m_CudaGraphicsResource);
        return reinterpret_cast<T*>(ptr);
    };

    template<typename T = uchar4>
    const T* GetConstDevicePtr() const {
        size_t size;
        void* ptr;
        cudaGraphicsResourceGetMappedPointer(&ptr, &size, m_CudaGraphicsResource);
        return reinterpret_cast<T*>(ptr);
    };

    GLuint GetTexture();

    void Resize(uint32_t a_Width, uint32_t a_Height);

    template<typename T = glm::vec3>
    T GetPixel(const glm::vec2 a_UVs)
    {
        if (a_UVs.x > 0.0f && a_UVs.x < 1.0f &&
            a_UVs.y > 0.0f && a_UVs.y < 1.0f)
        {
            glm::ivec2 pixelID;
            pixelID.x = a_UVs.x * m_Width;
            pixelID.y = a_UVs.y * m_Height;

            auto offset = pixelID.y * m_Width + pixelID.x;

            auto devPtr = GetDevicePtr<char>() + offset * m_PixelSize;

            T res;

            cudaMemcpy(&res, devPtr, std::min(m_PixelSize, static_cast<uint8_t>(sizeof(T))), cudaMemcpyKind::cudaMemcpyDeviceToHost);

            return res;
        }

        return T(0);
    }

    glm::ivec2 GetSize() const { return glm::ivec2(m_Width, m_Height); }

private:

    void UpdateTexture();

    uint32_t m_Width;
    uint32_t m_Height;
    uint8_t m_PixelSize;  

    GLuint m_PixelBuffer;
    GLuint m_Texture;
    GLuint m_Format;

    bool m_TextureDirty;

    cudaGraphicsResource* m_CudaGraphicsResource;

};