#include "InteropGPUTexture.h"
#include "CudaUtilities.h"

#include <memory>
#include <cassert>
#include <cuda_d3d11_interop.h>

using namespace Microsoft::WRL;

std::map<ComPtr<ID3D11Resource>, std::shared_ptr<cudaGraphicsResource_t>> InteropGPUTexture::s_RegisteredResources = {};
std::set<std::shared_ptr<cudaGraphicsResource_t>> InteropGPUTexture::s_MappedResources = {};

InteropGPUTexture::InteropGPUTexture(
    ComPtr<ID3D11Resource> a_TextureResource,
    const cudaTextureDesc& a_TextureDesc,
    unsigned int a_RegisterFlags)
    :
    m_Resource(nullptr),
    m_Array(nullptr),
    m_TextureDesc(a_TextureDesc),
    m_RegisterFlags(a_RegisterFlags),
    m_TextureObject(0),
    m_SurfaceObject(0)
{

    RegisterResource(a_TextureResource);

}

InteropGPUTexture::~InteropGPUTexture()
{

    

}

void InteropGPUTexture::Map(
    unsigned a_ArrayIndex, 
    unsigned a_MipLevel,
    unsigned a_MapFlags)
{

    assert(m_Resource != nullptr && "Resource was not initialized !");

    if(s_MappedResources.find(m_Resource) == s_MappedResources.end())
    {

        CHECKCUDAERROR(cudaGraphicsMapResources(1, m_Resource.get()));
        s_MappedResources.insert(m_Resource);

    }

    CHECKCUDAERROR(cudaGraphicsSubResourceGetMappedArray(
        &m_Array,
        *m_Resource,
        a_ArrayIndex,
        a_MipLevel));

}

void InteropGPUTexture::Unmap()
{

    const auto resourceIterator = s_MappedResources.find(m_Resource);
    if(resourceIterator != s_MappedResources.end())
    {

        CHECKCUDAERROR(cudaGraphicsUnmapResources(1, m_Resource.get()));
        s_MappedResources.erase(resourceIterator);

    }

    if (m_TextureObject)
    {
        CHECKCUDAERROR(cudaDestroyTextureObject(m_TextureObject));
        m_TextureObject = 0;
    }
    if (m_SurfaceObject)
    {
        CHECKCUDAERROR(cudaDestroySurfaceObject(m_SurfaceObject));
        m_SurfaceObject = 0;
    }
    if (m_Array)
    {
        m_Array = nullptr;
    }

}

void InteropGPUTexture::Clear() const
{

    assert(m_Array != nullptr && "Tried to clear an un-intialized array resource!");

    cudaChannelFormatDesc formatDesc{};
    memset(&formatDesc, 0, sizeof(formatDesc));

    cudaExtent extent{};
    memset(&extent, 0, sizeof(extent));

    unsigned int flags = 0;

    CHECKCUDAERROR(cudaArrayGetInfo(&formatDesc, &extent, &flags, m_Array));

    const unsigned formatByteSize = (formatDesc.x + formatDesc.y + formatDesc.z + formatDesc.w) / 8; //TotalNumber of bits, converted to number of bytes

    const unsigned int size =
        extent.width *
        std::max(extent.height, 1llu) *
        formatByteSize; 

    void* devPtr = nullptr;
    CHECKCUDAERROR(cudaMalloc(&devPtr, size));
    CHECKCUDAERROR(cudaMemset(devPtr, 0, size));

    CHECKCUDAERROR(cudaMemcpy2DToArray(
        m_Array, 
        0, 
        0, 
        devPtr, 
        extent.width * formatByteSize, 
        extent.width * formatByteSize, 
        extent.height, 
        cudaMemcpyDeviceToDevice));

    CHECKCUDAERROR(cudaFree(devPtr));

}

const cudaTextureObject_t& InteropGPUTexture::GetTextureObject()
{

    if (!m_TextureObject)
    {
        CreateTextureObject();
    }

    return m_TextureObject;

}

const cudaSurfaceObject_t& InteropGPUTexture::GetSurfaceObject()
{

    assert(m_RegisterFlags | cudaGraphicsRegisterFlagsSurfaceLoadStore &&
        "Tried to get surface object without initializing with appropriate flags");

    if (!m_SurfaceObject)
    {
        CreateSurfaceObject();
    }

    return m_SurfaceObject;

}

void InteropGPUTexture::RegisterResource(const ComPtr<ID3D11Resource>& a_TextureResource)
{

    const auto resourceIterator = s_RegisteredResources.find(a_TextureResource);
    if (resourceIterator != s_RegisteredResources.end())
    {
        m_Resource = resourceIterator->second;
        return;
    }
    else
    {

        cudaGraphicsResource_t tempPtr = nullptr;

        CHECKCUDAERROR(cudaGraphicsD3D11RegisterResource(&tempPtr, a_TextureResource.Get(), m_RegisterFlags));

        m_Resource = std::make_shared<cudaGraphicsResource_t>(tempPtr);

        s_RegisteredResources.insert({ a_TextureResource, m_Resource });
        
    }

}

void InteropGPUTexture::CreateTextureObject()
{

    assert(m_Array != nullptr && "Tried to create a texture object with an un-intialized array resource!");

    cudaResourceDesc resourceDesc{};
    memset(&resourceDesc, 0, sizeof(resourceDesc));

    resourceDesc.resType = cudaResourceTypeArray;
    resourceDesc.res.array.array = m_Array;

    CHECKCUDAERROR(cudaCreateTextureObject(&m_TextureObject, &resourceDesc, &m_TextureDesc, nullptr));

    return;

}

void InteropGPUTexture::CreateSurfaceObject()
{

    assert(m_Array != nullptr && "Tried to create a surface object with an un-initialized array resource!");

    cudaResourceDesc resourceDesc{};
    memset(&resourceDesc, 0, sizeof(resourceDesc));

    resourceDesc.resType = cudaResourceTypeArray;
    resourceDesc.res.array.array = m_Array;

    CHECKCUDAERROR(cudaCreateSurfaceObject(&m_SurfaceObject, &resourceDesc));

    return;

}