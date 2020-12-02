
#include "OutputBuffer.h"

#include "Cuda/cuda_gl_interop.h"

OutputBuffer::OutputBuffer(uint32_t a_Width, uint32_t a_Height)
{
    Resize(a_Width, a_Height);
}

OutputBuffer::~OutputBuffer()
{
    glDeleteBuffers(1, &m_PixelBuffer);
    glDeleteTextures(1, &m_Texture);
}

void* OutputBuffer::GetDevicePointer()
{
    size_t size;
    void* ptr;
    cudaGraphicsResourceGetMappedPointer(&ptr, &size, m_CudaGraphicsResource);

    return ptr;
}

GLuint OutputBuffer::GetTexture()
{
    UpdateTexture();
    return m_Texture;
}

void OutputBuffer::UpdateTexture()
{
    glBindTexture(GL_TEXTURE_2D, m_Texture);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_PixelBuffer);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_Width, m_Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}


void OutputBuffer::Resize(uint32_t a_Width, uint32_t a_Height)
{
    m_Width = a_Width;
    m_Height = a_Height;

    glGenBuffers(1, &m_PixelBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, m_PixelBuffer);
    glBufferData(GL_ARRAY_BUFFER, m_Width * m_Height * 4, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    cudaGraphicsGLRegisterBuffer(&m_CudaGraphicsResource, m_PixelBuffer, cudaGraphicsMapFlagsWriteDiscard);

    glGenTextures(1, &m_Texture);
}