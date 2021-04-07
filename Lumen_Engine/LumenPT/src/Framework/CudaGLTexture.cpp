
#include "CudaGLTexture.h"

#include "Cuda/cuda_gl_interop.h"

#include <cassert>

#include <stdio.h>
#include <iostream>
#include <vector>


CudaGLTexture::CudaGLTexture(GLuint a_Format, uint32_t a_Width, uint32_t a_Height, uint8_t a_PixelSize)
    : m_Format(a_Format)
    , m_PixelSize(a_PixelSize)
    , m_Texture(true)
{
    glGenBuffers(1, &m_PixelBuffer);
    glGenTextures(1, &m_Texture);
    Resize(a_Width, a_Height);
}

CudaGLTexture::~CudaGLTexture()
{
    glDeleteBuffers(1, &m_PixelBuffer);
    glDeleteTextures(1, &m_Texture);
}

GLuint CudaGLTexture::GetTexture()
{
    auto err = glGetError();

    UpdateTexture();
    return m_Texture;
}

void CudaGLTexture::UpdateTexture()
{
    if (m_TextureDirty)
    {
        GL_INVALID_OPERATION;
        auto err = glGetError();
        glBindTexture(GL_TEXTURE_2D, m_Texture);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_PixelBuffer);
        err = glGetError();
        glPixelStorei(GL_UNPACK_ALIGNMENT, m_PixelSize);
        auto format = GL_UNSIGNED_BYTE;
        auto pixelFormat = GL_RGBA;
        if (m_Format == GL_RGB32F)
        {
            format = GL_FLOAT;
            pixelFormat = GL_RGB;
        }
        glTexImage2D(GL_TEXTURE_2D, 0, m_Format, m_Width, m_Height, 0, pixelFormat, format, nullptr);
        err = glGetError();
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);


        if (pixelFormat == GL_RGBA)
        {
            err = glGetError();
            auto err1 = glGetError();
            auto err3 = glGetError();
            std::vector<uchar4> v(m_Width * m_Height);
            cudaMemcpy(v.data(), GetDevicePtr<void>(), v.size() * 4, cudaMemcpyDeviceToHost);
            err = glGetError();

            std::vector<uchar4> v1(m_Width * m_Height);
            err = glGetError();
            glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_BYTE, v1.data());

            glFlush();
            printf("copy");
        }

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        glBindTexture(GL_TEXTURE_2D, 0);
        m_TextureDirty = false;
    }
}


void CudaGLTexture::Resize(uint32_t a_Width, uint32_t a_Height)
{
    m_Width = a_Width;
    m_Height = a_Height;

    glBindBuffer(GL_ARRAY_BUFFER, m_PixelBuffer);
    glBufferData(GL_ARRAY_BUFFER, m_Width * m_Height * m_PixelSize, nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaDeviceSynchronize();
    if (m_Width != 0 && m_Height != 0)
    {
       cudaGetLastError();
       cudaGraphicsGLRegisterBuffer(&m_CudaGraphicsResource, m_PixelBuffer, cudaGraphicsMapFlagsWriteDiscard);
       cudaGraphicsMapResources(1, &m_CudaGraphicsResource);
    }
}