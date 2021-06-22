#include "InteropGPUTexture.h"
#include "CudaUtilities.h"

#include <memory>
#include <cassert>
#include <cuda_d3d11_interop.h>
#include <algorithm>

using namespace Microsoft::WRL;

std::map<
    const ComPtr<ID3D11Resource>,
    std::pair<
        std::shared_ptr<cudaGraphicsResource_t>,
        std::set<InteropGPUTexture*>
    >
> InteropGPUTexture::s_RegisteredResources = {};

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

bool InteropGPUTexture::RegisterResource(const ComPtr<ID3D11Resource>& a_Resource)
{

    //Check if the resource is already registered.
    const auto resourceIterator = s_RegisteredResources.find(a_Resource);

    //If the resource is already registered...
    if (resourceIterator != s_RegisteredResources.end())
    {

        std::set<InteropGPUTexture*>& registeredTextures = resourceIterator->second.second;

        //Check if this texture is not already registered.
        if (registeredTextures.find(this) == registeredTextures.end())
        {

            const auto insert = registeredTextures.insert(this); //Try to insert as a registered texture.
            if (insert.second)
            {
                m_Resource = resourceIterator->second.first;
            }
            else return false; //Texture has not been registered, failed to insert.
        }
        else return true; //Texture has already been registered, no need to register again.
    }
    else //If the resource is not registered yet...
    {

        
        cudaGraphicsResource_t resource = { nullptr };
        //Get a set of registered textures to a certain resource, initialize with current texture.
        const std::set<InteropGPUTexture*> registeredTextures = { this };

        //Register the resource.
        CHECKCUDAERROR(cudaGraphicsD3D11RegisterResource(&resource, a_Resource.Get(), m_RegisterFlags));

        //Store the resource as a shared_ptr for book keeping.
        std::shared_ptr<cudaGraphicsResource_t> resourcePtr = std::make_shared<cudaGraphicsResource_t>(resource);

        //Try to insert as a registered texture resource.
        const auto insert = s_RegisteredResources.insert({a_Resource, std::make_pair(resourcePtr, registeredTextures)});
        if(insert.second)
        {

            //Add shared_ptr reference to the texture resource.
            m_Resource = resourcePtr;
            
        }
    }

}

void InteropGPUTexture::UnRegisterResource(const ComPtr<ID3D11Resource>& a_Resource)
{

    //Check if the resource is registered.
    const auto resourceIterator = s_RegisteredResources.find(a_Resource);
    //If the resource if registered...
    if (resourceIterator != s_RegisteredResources.end())
    {

        //Unregister the resource.
        CHECKCUDAERROR(cudaGraphicsUnregisterResource(*resourceIterator->second.first));

        std::set<InteropGPUTexture*>& registeredTextures = resourceIterator->second.second;

        for (InteropGPUTexture* const texture : registeredTextures)
        {

            texture->m_Resource = nullptr;

        }

        s_RegisteredResources.erase(resourceIterator);
        return;

    }
    else return;
    //Resource is not unregistered, as it was not registered.
    //However, the goal is met, the resource is not registered (anymore) after this call.

}

bool InteropGPUTexture::Map(
    unsigned a_ArrayIndex, 
    unsigned a_MipLevel,
    unsigned a_MapFlags)
{

    assert(m_Resource != nullptr && "Resource was not initialized !");
    bool success = true;

    if(s_MappedResources.find(m_Resource) == s_MappedResources.end())
    {

        CHECKCUDAERROR(cudaGraphicsMapResources(1, m_Resource.get()));

        success = s_MappedResources.insert(m_Resource).second;

    }

    CHECKCUDAERROR(cudaGraphicsSubResourceGetMappedArray(
        &m_Array,
        *m_Resource,
        a_ArrayIndex,
        a_MipLevel));

    return success;

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