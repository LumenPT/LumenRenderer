#include "DX11Wrapper.h"
#include "DX11Utilties.h"

#include <cuda_d3d11_interop.h>
#include <cuda_runtime_api.h>

namespace WaveFront 
{
    void DX11Wrapper::Init()
    {
        UINT deviceFlags = 0;
    #if _DEBUG
        deviceFlags = D3D11_CREATE_DEVICE_DEBUG;
    #endif
        D3D_FEATURE_LEVEL featureLevels[] =
        {
            D3D_FEATURE_LEVEL_11_1,
            D3D_FEATURE_LEVEL_11_0,
            D3D_FEATURE_LEVEL_10_1,
            D3D_FEATURE_LEVEL_10_0,
            D3D_FEATURE_LEVEL_9_3,
            D3D_FEATURE_LEVEL_9_2,
            D3D_FEATURE_LEVEL_9_1
        };

        m_D3dDevice = nullptr;
        m_D3dDeviceContext = nullptr;
        D3D_FEATURE_LEVEL featureLevel;

        HRESULT hr = D3D11CreateDevice(
            nullptr, D3D_DRIVER_TYPE_HARDWARE,
            nullptr, deviceFlags, featureLevels,
            _countof(featureLevels), D3D11_SDK_VERSION,
            &m_D3dDevice, &featureLevel, &m_D3dDeviceContext);

    }

    Microsoft::WRL::ComPtr<ID3D11Texture2D> DX11Wrapper::CreateTexture2D(const uint3& a_ResDepth)
    {

        DXGI_SAMPLE_DESC textureSampleDesc{};
        textureSampleDesc.Count = 1;
        textureSampleDesc.Quality = 0;

        D3D11_TEXTURE2D_DESC desc{};
        desc.Width = a_ResDepth.x;
        desc.Height = a_ResDepth.y;
        desc.ArraySize = a_ResDepth.z;
        desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT; //TODO: find out how to use 16 bit float format with CUDA (DLSS & NRD requirement). Provide as param?
        desc.SampleDesc = textureSampleDesc;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = 0;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;
        desc.MiscFlags = 0;
        // just create comptr to texture2D from (pixel) any buffer (?)
        Microsoft::WRL::ComPtr<ID3D11Texture2D> tex {nullptr};
        CHECKDX11RESULT(m_D3dDevice->CreateTexture2D(&desc, nullptr, &tex));
        return tex;

    }

    Microsoft::WRL::ComPtr<ID3D11Texture2D> DX11Wrapper::ResizeTexture2D(Microsoft::WRL::ComPtr<ID3D11Texture2D>& a_Tex, const uint2& a_NewSize)
    {
        D3D11_TEXTURE2D_DESC desc;
        a_Tex->GetDesc(&desc);
        desc.Width = a_NewSize.x;
        desc.Height = a_NewSize.y;
        CHECKDX11RESULT(m_D3dDevice->CreateTexture2D(&desc, nullptr, &a_Tex)); //Data is lost now
        return a_Tex;
    }

}