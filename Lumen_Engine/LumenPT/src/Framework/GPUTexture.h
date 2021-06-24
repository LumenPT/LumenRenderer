#pragma once 
#include <cuda_runtime.h> 
#include <initializer_list> 

template<typename T>
class GPUTexture
{

public:

    /// <summary> 
    /// Creates a Cuda texture. 
    /// </summary> 
    /// <param name="a_Extent"></param> 
    /// <param name="a_Flags">Flags to use to create the texture. Can be used to make a layered texture/array or/and a cube-map.</param> 
    /// <param name="a_TextureDesc"></param> 
    /// <param name="a_Data">Optional. Allows data to be passed for uploading when creating the texture.</param> 
    GPUTexture(
        cudaExtent a_Extent = { 0, 0, 0 },
        unsigned a_Flags = cudaArrayDefault,
        const cudaTextureDesc& a_TextureDesc = {},
        T* a_Data = nullptr);

    ~GPUTexture();

    void Write(
        cudaPitchedPtr a_Src,
        cudaPos a_SrcPos,
        cudaPos a_DesPos,
        cudaExtent a_Extent) const;

    void Read(
        cudaPitchedPtr a_Des,
        cudaPos a_SrcPos,
        cudaPos a_DesPos,
        cudaExtent a_Extent) const;

    void Clear() const;

    void Resize(cudaExtent a_Extent);

    const cudaTextureObject_t& GetTextureObject() const;

    const cudaSurfaceObject_t& GetSurfaceObject() const;

    const cudaExtent& GetExtent() const;

private:

    void CreateTexture(
        const cudaTextureDesc& a_TextureDesc,
        T* a_Data);

    cudaExtent m_Extent;
    unsigned m_Flags;

    cudaArray_t m_TextureMemory;
    cudaTextureObject_t m_TextureObject;
    cudaSurfaceObject_t m_SurfaceObject;

};

#include "GPUTexture.cpp"
