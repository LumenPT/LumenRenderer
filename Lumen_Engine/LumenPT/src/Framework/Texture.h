#pragma once

#include "Cuda/cuda_runtime_api.h"

#include <cstdint>
#include <string>

class Texture
{
public:

    Texture(std::string a_Path);

    Texture(void* a_PixelData, cudaChannelFormatDesc a_FormatDesc, uint32_t a_Width, uint32_t a_Height);

    cudaTextureObject_t& operator*() { return m_TextureObject; }

private:

    void CreateTextureObject(void* a_PixelData, cudaChannelFormatDesc a_FormatDesc, uint32_t a_Width, uint32_t a_Height);

    uint32_t m_Width;
    uint32_t m_Height;

    cudaTextureObject_t m_TextureObject;
    cudaArray_t m_MemoryArray;
};

