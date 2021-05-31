#include "DX11Wrapper.h"

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

    ID3D11Device* g_d3dDevice = nullptr;
    ID3D11DeviceContext* g_d3dDeviceContext = nullptr;
    D3D_FEATURE_LEVEL featureLevel;

    HRESULT hr = D3D11CreateDevice(
        nullptr, D3D_DRIVER_TYPE_HARDWARE,
        nullptr, deviceFlags, featureLevels,
        _countof(featureLevels), D3D11_SDK_VERSION,
        &g_d3dDevice, &featureLevel, &g_d3dDeviceContext);

}
