#pragma once

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>


#include "Cuda/cuda_runtime.h"
#include "Glad/glad.h"

#include "cstdint"
#include <algorithm>


// Class which serves as a bridge between CUDA/OptiX and OpenGL for image output purposes of any sorts.
class CudaGLTexture
{
public:
    // Defaults to 0x0 texture of four 8-bit channels
    CudaGLTexture(GLuint a_Format = GL_RGBA8, uint32_t a_Width = 0, uint32_t a_Height = 0, uint8_t a_PixelSize = 4);
    ~CudaGLTexture();

    // Returns a CUDA-usable GPU pointer of the specified template type.
    // Expects the texture data to be initialized beforehand, else expect CUDA errors when using the resulting pointer.
    template<typename T = uchar4>
    T* GetDevicePtr() {
        m_TextureDirty = true;
        Map();
        return reinterpret_cast<T*>(m_CudaPtr);
    };

    template<typename T = uchar4>
    const T* GetConstDevicePtr() const {
        Map();
        return reinterpret_cast<T*>(m_CudaPtr);
    };

    // Returns the OpenGL texture handle for use in output pipeline or ImGui.
    GLuint GetTexture();

    // Resize the texture and pixel buffer to fit an image of the provided dimensions, given the assigned pixel format and pixel size
    void Resize(uint32_t a_Width, uint32_t a_Height);

    // Returns the content of the pixel to the provided UV coordinates, returned as the template parameter type.
    // If the pixel size of the texture is smaller than the requested output type, only a single pixel will be output.
    // Returns 0 if either of the UVs are outside of the [0.0, 1.0] range
    template<typename T = glm::vec3>
    T GetPixel(const glm::vec2 a_UVs)
    {
        if (a_UVs.x > 0.0f && a_UVs.x < 1.0f &&
            a_UVs.y > 0.0f && a_UVs.y < 1.0f)
        {
            // Determine the ID of the pixel at the specified UV coordinates
            glm::ivec2 pixelID;
            pixelID.x = a_UVs.x * m_Width;
            pixelID.y = a_UVs.y * m_Height;

            // Calculate the offset into the buffer based on the pixel index, and calculate the pointer to read from
            auto offset = pixelID.y * m_Width + pixelID.x;
            auto devPtr = GetDevicePtr<char>() + offset * m_PixelSize;

            // Copy from the GPU to the CPU. If sizeof(T) is bigger than the pixel size of the texture, only m_PixelSize bytes are copied to avoid
            // copying from illegal memory addresses when trying to sample a_UV = (1.0, 1.0)
            T res;

            cudaMemcpy(&res, devPtr, std::min(m_PixelSize, static_cast<uint8_t>(sizeof(T))), cudaMemcpyKind::cudaMemcpyDeviceToHost);

            return res;
        }

        return T(0);
    }

    // Returns the dimensions of the texture
    glm::ivec2 GetSize() const { return glm::ivec2(m_Width, m_Height); }

    void Map() const;
    void Unmap() const;
private:

    void UpdateTexture();


    mutable void* m_CudaPtr;
    mutable cudaGraphicsResource* m_CudaGraphicsResource; // Cuda handle to allow using the OpenGL pixel buffer with CUDA.

    uint32_t m_Width; // PTTexture Width
    uint32_t m_Height; // PTTexture Height
    uint8_t m_PixelSize; // Size of the pixels in bytes

    GLuint m_PixelBuffer; // OpenGL handle to a buffer resource that is used as a pixel buffer for CUDA
    GLuint m_Texture; // OpenGL handle to a texture that is used when displaying the pixel buffer
    GLuint m_Format; // The OpenGL format to use for the texture. Defaults to GL_RGBA8

    bool m_TextureDirty;

};