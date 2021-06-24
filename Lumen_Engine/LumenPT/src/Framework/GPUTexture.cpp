#ifndef __GPUTEXTURE_CPP__ 
#define __GPUTEXTURE_CPP__ 
#include "GPUTexture.h" 
#include "CudaUtilities.h" 

#include <memory> 
#include <cassert> 

template <typename T>
GPUTexture<T>::GPUTexture(
    cudaExtent a_Extent,
    unsigned a_Flags,
    const cudaTextureDesc& a_TextureDesc,
    T* a_Data)
    :
    m_Extent(a_Extent),
    m_Flags(a_Flags),
    m_TextureMemory(nullptr),
    m_TextureObject(0),
    m_SurfaceObject(0)
{

    if (m_Extent.width == 0 &&
        m_Extent.height == 0 &&
        m_Extent.depth == 0)
    {
        return;
    }

    //TODO: ensure that the size requested does not exceed device capabilities. 
    this->CreateTexture(a_TextureDesc, a_Data);

}

template <typename T>
GPUTexture<T>::~GPUTexture()
{

    cudaFreeArray(m_TextureMemory);
    cudaDestroySurfaceObject(m_SurfaceObject);
    cudaDestroyTextureObject(m_TextureObject);

}


template <typename T>
void GPUTexture<T>::CreateTexture(
    const cudaTextureDesc& a_TextureDesc,
    T* a_Data)
{

    cudaChannelFormatDesc format = cudaCreateChannelDesc<T>();
    CHECKCUDAERROR(cudaMalloc3DArray(&m_TextureMemory, &format, m_Extent, m_Flags));

    cudaResourceDesc textureResDesc{};
    memset(&textureResDesc, 0, sizeof(textureResDesc));

    textureResDesc.resType = cudaResourceTypeArray;
    textureResDesc.res.array.array = m_TextureMemory;

    CHECKCUDAERROR(cudaCreateTextureObject(&m_TextureObject, &textureResDesc, &a_TextureDesc, nullptr));

    if (m_Flags & cudaArraySurfaceLoadStore)
    {
        CHECKCUDAERROR(cudaCreateSurfaceObject(&m_SurfaceObject, &textureResDesc));
    }

    if (a_Data)
    {

        cudaPitchedPtr src{ nullptr };
        src.ptr = a_Data;
        src.pitch = m_Extent.width * sizeof(T);
        src.xsize = m_Extent.width;
        src.ysize = m_Extent.height;

        const cudaPos pos{ 0, 0, 0 };

        this->Write(src, pos, pos, m_Extent);

    }

}

template <typename T>
void GPUTexture<T>::Write(
    cudaPitchedPtr a_Src,
    cudaPos a_SrcPos,
    cudaPos a_DesPos,
    cudaExtent a_Extent) const
{

    cudaMemcpy3DParms copyParams{};
    memset(&copyParams, 0, sizeof(copyParams));

    copyParams.srcPos = a_SrcPos;
    copyParams.srcPtr = a_Src;
    copyParams.dstArray = m_TextureMemory;
    copyParams.dstPos = a_DesPos;
    copyParams.extent = a_Extent;
    copyParams.kind = cudaMemcpyHostToDevice;

    CHECKCUDAERROR(cudaMemcpy3D(&copyParams));

}

template <typename T>
void GPUTexture<T>::Read(
    cudaPitchedPtr a_Des,
    cudaPos a_SrcPos,
    cudaPos a_DesPos,
    cudaExtent a_Extent) const
{

    cudaMemcpy3DParms copyParams{};
    memset(&copyParams, 0, sizeof(copyParams));

    copyParams.srcArray = m_TextureMemory;
    copyParams.srcPos = a_SrcPos;
    copyParams.dstPos = a_DesPos;
    copyParams.dstPtr = a_Des;
    copyParams.extent = a_Extent;
    copyParams.kind = cudaMemcpyDeviceToHost;

    CHECKCUDAERROR(cudaMemcpy3D(&copyParams));

}

template <typename T>
void GPUTexture<T>::Clear() const
{

    const unsigned int size =
        m_Extent.width *
        std::max(m_Extent.height, 1llu) *
        std::max(m_Extent.depth, 1llu) *
        sizeof(T);

    void* devPtr = nullptr;
    CHECKCUDAERROR(cudaMalloc(&devPtr, size));
    CHECKCUDAERROR(cudaMemset(devPtr, 0, size));

    cudaPitchedPtr srcPtr{ nullptr };
    srcPtr.ptr = devPtr;
    srcPtr.pitch = m_Extent.width * sizeof(T);
    srcPtr.xsize = m_Extent.width;
    srcPtr.ysize = m_Extent.height;

    const cudaPos pos{ 0,0,0 };

    cudaMemcpy3DParms copyParams{};
    memset(&copyParams, 0, sizeof(copyParams));
    copyParams.srcPtr = srcPtr;
    copyParams.srcPos = pos;
    copyParams.dstArray = m_TextureMemory;
    copyParams.dstPos = pos;
    copyParams.extent = m_Extent;
    copyParams.kind = cudaMemcpyDeviceToDevice;

    CHECKCUDAERROR(cudaMemcpy3D(&copyParams));
    CHECKCUDAERROR(cudaFree(devPtr));

}

template <typename T>
void GPUTexture<T>::Resize(cudaExtent a_Extent)
{

    cudaTextureDesc textureDescription{};
    memset(&textureDescription, 0, sizeof(textureDescription));

    if (m_SurfaceObject) { CHECKCUDAERROR(cudaDestroySurfaceObject(m_SurfaceObject)); }

    if (m_TextureObject)
    {
        CHECKCUDAERROR(cudaGetTextureObjectTextureDesc(&textureDescription, m_TextureObject));
        CHECKCUDAERROR(cudaDestroyTextureObject(m_TextureObject));
    }

    if (m_TextureMemory) { CHECKCUDAERROR(cudaFreeArray(m_TextureMemory)); }

    m_Extent = a_Extent;

    if (m_Extent.width == 0 &&
        m_Extent.height == 0 &&
        m_Extent.depth == 0)
    {
        return;
    }

    this->CreateTexture(textureDescription, nullptr);

}



template <typename T>
const cudaTextureObject_t& GPUTexture<T>::GetTextureObject() const
{

    assert(m_TextureObject != 0 && "Texture object is null");

    return m_TextureObject;

}

template <typename T>
const cudaSurfaceObject_t& GPUTexture<T>::GetSurfaceObject() const
{

    assert(m_SurfaceObject != 0 && "Surface object is null");

    return m_SurfaceObject;

}

#endif