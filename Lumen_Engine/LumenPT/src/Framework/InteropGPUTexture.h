#pragma once
#include <cuda_runtime.h>
#include <wrl.h>
#include <d3d11.h>
#include <memory>
#include <map>
#include <set>

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

class InteropGPUTexture
{

public:

    template<typename PtrT>
    using ComPtr = Microsoft::WRL::ComPtr<PtrT>;

    InteropGPUTexture(
        ComPtr<ID3D11Resource> a_TextureResource,
        const cudaTextureDesc& a_TextureDesc,
        unsigned int a_RegisterFlags = cudaGraphicsRegisterFlagsNone);

    ~InteropGPUTexture();

    bool RegisterResource(const ComPtr<ID3D11Resource>& a_Resource);

    static void UnRegisterResource(const ComPtr<ID3D11Resource>& a_Resource);

    bool Map(
        unsigned int a_ArrayIndex = 0,
        unsigned int a_MipLevel = 0,
        unsigned int a_MapFlags = cudaGraphicsMapFlagsNone);

    void Unmap();

    void Clear() const;

    const cudaTextureObject_t& GetTextureObject();
    const cudaSurfaceObject_t& GetSurfaceObject();

private:

    void CreateTextureObject();

    void CreateSurfaceObject();

    std::shared_ptr<cudaGraphicsResource_t> m_Resource;
    cudaArray_t m_Array;

    cudaTextureDesc m_TextureDesc;
    unsigned int m_RegisterFlags;

    cudaTextureObject_t m_TextureObject;
    cudaSurfaceObject_t m_SurfaceObject;

    //TODO: maybe needs a different container types for performance, Registering only happens during creation of the buffer and resizing, Mapping happens every frame.
    static std::map<const ComPtr<ID3D11Resource>,
        std::pair<
            std::shared_ptr<cudaGraphicsResource_t>,
            std::set<InteropGPUTexture*>>> s_RegisteredResources;

    static std::set<std::shared_ptr<cudaGraphicsResource_t>> s_MappedResources; 

};