#pragma once
#pragma comment(lib,"d3d11.lib")

#include <d3d11.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>
namespace WaveFront
{
	class DX11Wrapper
	{
	public:
		void Init();
		inline ID3D11Device* GetDevice() { return m_D3dDevice; };
		inline ID3D11DeviceContext* GetContext() { return m_D3dDeviceContext; };

	private:
		ID3D11Device* m_D3dDevice = nullptr;
		ID3D11DeviceContext* m_D3dDeviceContext = nullptr;

	};
}