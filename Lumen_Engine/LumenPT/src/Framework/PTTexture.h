#pragma once

#include "Cuda/cuda_runtime_api.h"

#include "Renderer/ILumenResources.h"

#include <cstdint>
#include <string>


// Path tracer specific texture class to account for implementation details connected to the APIs used.
class PTTexture : public Lumen::ILumenTexture
{
public:

    // Create a texture from a given file path. Uses stb_image, so most common image format are supported.
    PTTexture(std::string a_Path, bool a_Normalize);

    // Create a texture from raw pixel data
    PTTexture(void* a_PixelData, cudaChannelFormatDesc& a_FormatDesc, uint32_t a_Width, uint32_t a_Height, bool a_Normalize);

    ~PTTexture();

    // Returns the underlying cudaTextureObject_t that is used to sample the texture in GPU shaders
    cudaTextureObject_t& operator*() { return m_TextureObject; }

private:

    void CreateTextureObject(void* a_PixelData, cudaChannelFormatDesc& a_FormatDesc, uint32_t a_Width, uint32_t a_Height, bool a_Normalize);

    uint32_t m_Width; // Texture width
    uint32_t m_Height; // Texture height

    // Textures in CUDA are represented by a cudaTextureObject_t handle.
    // This is also used when sampling from the GPU. This does not have any inherent memory allocated in it.
    cudaTextureObject_t m_TextureObject;
    // The memory used by the texture
    cudaArray_t m_MemoryArray;
};

