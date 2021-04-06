#include "PTTexture.h"

#include "AssetLoading.h"

#include "cuda_runtime.h"

#include <cassert>

PTTexture::PTTexture(std::string a_Path)
{
    // Load the data using stb_image
    int x, y, c;
    auto data = stbi_load(a_Path.c_str(), &x, &y, &c, 4);

    // Create a texture from the loaded data
    CreateTextureObject(data, cudaCreateChannelDesc<uchar4>(), x, y);

    // Free the loaded data to avoid cramming up the CPU memory
    stbi_image_free(data);

}

PTTexture::PTTexture(void* a_PixelData, cudaChannelFormatDesc& a_FormatDesc, uint32_t a_Width, uint32_t a_Height)
    : m_Width(a_Width)
    , m_Height(a_Height)
{
    CreateTextureObject(a_PixelData, a_FormatDesc, a_Width, a_Height);
}

PTTexture::~PTTexture()
{
    cudaFreeArray(m_MemoryArray);
}

void PTTexture::CreateTextureObject(void* a_PixelData, cudaChannelFormatDesc& a_FormatDesc, uint32_t a_Width, uint32_t a_Height)
{
    m_Width = a_Width;
    m_Height = a_Height;

    // Allocate the memory for the texture
    cudaMallocArray(&m_MemoryArray, &a_FormatDesc, m_Width, m_Height);

    auto pixelSize = a_FormatDesc.x + a_FormatDesc.y + a_FormatDesc.z + a_FormatDesc.w;

    // The initial result is in bits, we want to know the size in bytes
    pixelSize /= 8;

    // Copy the raw data to the allocated memory with a CPU to GPU copy
    cudaMemcpy2DToArray(m_MemoryArray, 0, 0, a_PixelData, a_Width * pixelSize, a_Width * pixelSize, a_Height, cudaMemcpyHostToDevice);

    // Describe the resource memory of the texture
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = m_MemoryArray;

    // Describe the sampling of the texture
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeWrap; // Wrap around on U
    texDesc.addressMode[1] = cudaAddressModeWrap; // Wrap around on V
    texDesc.addressMode[2] = cudaAddressModeWrap; // Wrap around on W
    texDesc.filterMode = cudaFilterModeLinear; 
    texDesc.minMipmapLevelClamp = 0.0f;
    texDesc.maxMipmapLevelClamp = 99.0f;
    texDesc.mipmapLevelBias = 0.0f;
    texDesc.mipmapFilterMode = cudaFilterModeLinear;
    texDesc.normalizedCoords = 1;
    texDesc.maxAnisotropy = 1;
    texDesc.disableTrilinearOptimization = 1;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.sRGB = 1;

    // Finally, create the texture object itself
    cudaCreateTextureObject(&m_TextureObject, &resDesc, &texDesc, nullptr);
}
