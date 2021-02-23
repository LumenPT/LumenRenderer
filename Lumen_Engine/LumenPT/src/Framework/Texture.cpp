#include "Texture.h"

#include "AssetLoading.h"

#include "cuda_runtime.h"

#include <cassert>

void CudaCheck1(cudaError_t a_Err)
{
    assert(a_Err == cudaSuccess);
}

Texture::Texture(std::string a_Path)
{
    int x, y, c;
    auto data = stbi_load(a_Path.c_str(), &x, &y, &c, 4);

    CreateTextureObject(data, cudaCreateChannelDesc<uchar4>(), x, y);

    stbi_image_free(data);

}

Texture::Texture(void* a_PixelData, cudaChannelFormatDesc& a_FormatDesc, uint32_t a_Width, uint32_t a_Height)
    : m_Width(a_Width)
    , m_Height(a_Height)
{
    CreateTextureObject(a_PixelData, a_FormatDesc, a_Width, a_Height);
}

Texture::~Texture()
{
    cudaFreeArray(m_MemoryArray);
}

void Texture::CreateTextureObject(void* a_PixelData, cudaChannelFormatDesc& a_FormatDesc, uint32_t a_Width, uint32_t a_Height)
{
    m_Width = a_Width;
    m_Height = a_Height;

    CudaCheck1(cudaMallocArray(&m_MemoryArray, &a_FormatDesc, m_Width, m_Height));

    auto pixelSize = a_FormatDesc.x + a_FormatDesc.y + a_FormatDesc.z + a_FormatDesc.w;

    pixelSize /= 8;

    CudaCheck1(cudaMemcpy2DToArray(m_MemoryArray, 0, 0, a_PixelData, a_Width * pixelSize, a_Width * pixelSize, a_Height, cudaMemcpyHostToDevice));

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = m_MemoryArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.addressMode[2] = cudaAddressModeWrap;
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

    CudaCheck1(cudaCreateTextureObject(&m_TextureObject, &resDesc, &texDesc, nullptr));
}
