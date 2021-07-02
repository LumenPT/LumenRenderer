#include "DX11Wrapper.h"
#include "DX11Utilties.h"

#include <cuda_d3d11_interop.h>
#include <cuda_runtime_api.h>

namespace WaveFront 
{
    void DX11Wrapper::Init()
    {
        UINT deviceFlags = 0;

        deviceFlags = D3D11_CREATE_DEVICE_DEBUG;

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

    Microsoft::WRL::ComPtr<ID3D11Texture2D> DX11Wrapper::CreateTexture2D(const uint3& a_ResDepth, DXGI_FORMAT a_Format, UINT a_BindFlag, UINT a_Usage)
    {
        
        DXGI_SAMPLE_DESC textureSampleDesc{};
        textureSampleDesc.Count = 1;
        textureSampleDesc.Quality = 0;

        D3D11_USAGE usage = a_Usage == 0 ? D3D11_USAGE_DEFAULT : D3D11_USAGE_STAGING;

        D3D11_TEXTURE2D_DESC desc{};
        desc.Width = a_ResDepth.x;
        desc.Height = a_ResDepth.y;
        desc.MipLevels = 1;
        desc.ArraySize = a_ResDepth.z;
        desc.Format = a_Format;
        desc.SampleDesc = textureSampleDesc;
        //desc.Usage = D3D11_USAGE_DEFAULT;
        desc.Usage = usage;
        desc.BindFlags = a_BindFlag;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;
        desc.MiscFlags = 0;
        // just create comptr to texture2D from (pixel) any buffer (?)
        Microsoft::WRL::ComPtr<ID3D11Texture2D> tex {nullptr};
        CHECKDX11RESULT(m_D3dDevice->CreateTexture2D(&desc, nullptr, &tex));
        return tex;

    }

    Microsoft::WRL::ComPtr<ID3D11Texture2D> DX11Wrapper::ResizeTexture2D(Microsoft::WRL::ComPtr<ID3D11Texture2D>& a_Tex, const uint3& a_NewSize)
    {
        D3D11_TEXTURE2D_DESC desc;
        a_Tex->GetDesc(&desc);

        if( a_NewSize.x != desc.Width   ||
            a_NewSize.y != desc.Height  ||
            a_NewSize.z != desc.ArraySize)
        {

            desc.Width = a_NewSize.x;
            desc.Height = a_NewSize.y;
            desc.ArraySize = a_NewSize.z;
            CHECKDX11RESULT(m_D3dDevice->CreateTexture2D(&desc, nullptr, a_Tex.ReleaseAndGetAddressOf())); //Data is lost 
            
        }

        return a_Tex;
        
    }

    void DX11Wrapper::CreateUAV(Microsoft::WRL::ComPtr<ID3D11Resource> a_Res, const D3D11_UNORDERED_ACCESS_VIEW_DESC* a_UAVDesc, Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> a_UAVPtr)
    {
        CHECKDX11RESULT(m_D3dDevice->CreateUnorderedAccessView(a_Res.Get(), a_UAVDesc, a_UAVPtr.GetAddressOf()));
    }
}