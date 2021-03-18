#pragma once

#include "Cuda/cuda_runtime.h"
#include "Glad/glad.h"

#include "cstdint"

class OutputBuffer
{
public:
    OutputBuffer(uint32_t a_Width, uint32_t a_Height);
    OutputBuffer();
    ~OutputBuffer();

    uchar4* GetDevicePointer();

    GLuint GetTexture();

    void Resize(uint32_t a_Width, uint32_t a_Height);

private:

    void UpdateTexture();

    uint32_t m_Width;
    uint32_t m_Height;

    GLuint m_PixelBuffer;
    GLuint m_Texture;

    cudaGraphicsResource* m_CudaGraphicsResource;

};