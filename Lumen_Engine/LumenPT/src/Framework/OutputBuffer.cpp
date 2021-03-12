
#include "OutputBuffer.h"

#include "Cuda/cuda_gl_interop.h"

#include <cassert>

#include <stdio.h>
#include <iostream>

#ifndef _NDEBUG
#define GLCHECK(call)                      \
    do {                                   \
        call;                              \
        auto err = glGetError();           \
                                           \
        if (err != 0)    {                 \
            printf("GL error %d", err);    \
            abort();  }                    \
        assert(glGetError() == 0);         \
    } while(0)                             
#else
#define GLCHECK(void) 
#endif

OutputBuffer::OutputBuffer(uint32_t a_Width, uint32_t a_Height)
{
    Resize(a_Width, a_Height);
}

OutputBuffer::OutputBuffer()
{

}

OutputBuffer::~OutputBuffer()
{
    glDeleteBuffers(1, &m_PixelBuffer);
    glDeleteTextures(1, &m_Texture);
}

uchar4* OutputBuffer::GetDevicePointer()
{
    size_t size;
    void* ptr;
    auto err = cudaGraphicsResourceGetMappedPointer(&ptr, &size, m_CudaGraphicsResource);
    err;
    return reinterpret_cast<uchar4*>(ptr);
}

GLuint OutputBuffer::GetTexture()
{
    UpdateTexture();
    return m_Texture;
}

void OutputBuffer::UpdateTexture()
{
    GLCHECK(glBindTexture(GL_TEXTURE_2D, m_Texture));
    GLCHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_PixelBuffer));

    GLCHECK(glPixelStorei(GL_UNPACK_ALIGNMENT, 4));

    GLCHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_Width, m_Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr));

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    GLCHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));
    GLCHECK(glBindTexture(GL_TEXTURE_2D, 0));
}


void OutputBuffer::Resize(uint32_t a_Width, uint32_t a_Height)
{
    m_Width = a_Width;
    m_Height = a_Height;

    GLCHECK(glGenBuffers(1, &m_PixelBuffer));
    GLCHECK(glBindBuffer(GL_ARRAY_BUFFER, m_PixelBuffer));
    GLCHECK(glBufferData(GL_ARRAY_BUFFER, m_Width * m_Height * 4, nullptr, GL_STREAM_DRAW));
    GLCHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));
    cudaGraphicsGLRegisterBuffer(&m_CudaGraphicsResource, m_PixelBuffer, cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsMapResources(1, &m_CudaGraphicsResource);
    glGenTextures(1, &m_Texture);
}