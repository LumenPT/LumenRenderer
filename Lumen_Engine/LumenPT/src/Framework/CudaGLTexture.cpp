
#include "CudaGLTexture.h"

#include "Cuda/cuda_gl_interop.h"

#include <cassert>

#include <stdio.h>
#include <iostream>


CudaGLTexture::CudaGLTexture(uint32_t a_Width, uint32_t a_Height)
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

uchar4* CudaGLTexture::GetDevicePointer()
{
    size_t size;
    void* ptr;
    auto err = cudaGraphicsResourceGetMappedPointer(&ptr, &size, m_CudaGraphicsResource);
    err;
    return reinterpret_cast<uchar4*>(ptr);
}

GLuint CudaGLTexture::GetTexture()
{
    UpdateTexture();
    return m_Texture;
}

void CudaGLTexture::UpdateTexture()
{
    glBindTexture(GL_TEXTURE_2D, m_Texture);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_PixelBuffer);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_Width, m_Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}


void CudaGLTexture::Resize(uint32_t a_Width, uint32_t a_Height)
{
    m_Width = a_Width;
    m_Height = a_Height;

    glBindBuffer(GL_ARRAY_BUFFER, m_PixelBuffer);
    glBufferData(GL_ARRAY_BUFFER, m_Width * m_Height * 4, nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&m_CudaGraphicsResource, m_PixelBuffer, cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsMapResources(1, &m_CudaGraphicsResource);
}