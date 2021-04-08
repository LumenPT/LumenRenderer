
#include "CudaGLTexture.h"

#include "Cuda/cuda_gl_interop.h"

#include "Lumen/GLTaskSystem.h"

#include <cassert>

#include <stdio.h>
#include <iostream>
#include <vector>


CudaGLTexture::CudaGLTexture(GLuint a_Format, uint32_t a_Width, uint32_t a_Height, uint8_t a_PixelSize)
    : m_Format(a_Format)
    , m_PixelSize(a_PixelSize)
    , m_Texture(true)
    , m_CudaPtr(nullptr)
{
    // The pixel buffer and texture handles are generated only once during construction
    // Afterwards they are only repopulated with different data when necessary
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
    //Unmap();
    // The texture is updated every time it is requested
    // TODO: Limit the update frequency to only when the pixel buffer was changed.
    UpdateTexture();

    return m_Texture;
}

void CudaGLTexture::UpdateTexture()
{
    if (m_TextureDirty)
    {
        
        // To copy from a pixel buffer into a texture, we first need to bind both of them
        // The buffer is bound as GL_PIXEL_UNPACK_BUFFER
        glBindTexture(GL_TEXTURE_2D, m_Texture);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_PixelBuffer);

        // Specify the size of the pixels in the pixel buffer
        glPixelStorei(GL_UNPACK_ALIGNMENT, m_PixelSize);

        // Hacky way to ensure the texture is created with correct OpenGL formats
        // TODO: Improve 
        auto format = GL_UNSIGNED_BYTE;
        auto pixelFormat = GL_RGBA;
        if (m_Format == GL_RGB32F)
        {
            format = GL_FLOAT;
            pixelFormat = GL_RGB;
        }
        // Using glTextImage2D with a bound pixel buffer will copy from the pixel buffer rather than from the provided CPU pointer
        // In this scenario, the CPU pointer is interpreted as a 64bit offset into the pixel buffer to determine from where to start the copying operations
        glTexImage2D(GL_TEXTURE_2D, 0, m_Format, m_Width, m_Height, 0, pixelFormat, format, nullptr);

        // Set the sampler settings, standard procedure
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        // Unbind the pixel buffer and texture to avoid mistakes with future OpenGL calls
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        glBindTexture(GL_TEXTURE_2D, 0);
        m_TextureDirty = false;
    }
}

void CudaGLTexture::Unmap() const
{
    if (m_CudaPtr)
    {
        cudaGraphicsUnmapResources(1, &m_CudaGraphicsResource);
        m_CudaPtr = nullptr;
    }
}

void CudaGLTexture::Map() const
{
    if (!m_CudaPtr)
    {
        auto err = cudaGraphicsMapResources(1, &m_CudaGraphicsResource);
        size_t size;
        err = cudaGraphicsResourceGetMappedPointer(&m_CudaPtr, &size, m_CudaGraphicsResource);
    }
}


void CudaGLTexture::Resize(uint32_t a_Width, uint32_t a_Height)
{
    Unmap();
    m_Width = a_Width;
    m_Height = a_Height;

    // Allocate enough GPU memory through the OpenGL pixel buffer
    // glBufferData will only allocate the memory without filling it in if no CPU pointer is provided
    glBindBuffer(GL_ARRAY_BUFFER, m_PixelBuffer);
    glBufferData(GL_ARRAY_BUFFER, m_Width * m_Height * m_PixelSize, nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    // If any of the dimensions are 0 or less, the allocated memory can cause issues with CUDA, so we only continue with CUDA if
    // we are sure that the pixel buffer size will not be 0
    if (m_Width > 0 && m_Height > 0)
    {
        // Register the pixel buffer with CUDA, essentially allowing CUDA to work with it
        // cudaGraphicsMapFlagsWriteDiscard specifies that CUDA will exclusively write to the buffer, so its previous contents are discarded
        cudaGraphicsGLRegisterBuffer(&m_CudaGraphicsResource, m_PixelBuffer, cudaGraphicsMapFlagsWriteDiscard);        
        Map();
    }
}
