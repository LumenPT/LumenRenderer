#include "DX11Wrapper.h"
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

}