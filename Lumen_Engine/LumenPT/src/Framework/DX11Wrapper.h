#pragma once
#pragma comment(lib,"d3d11.lib")

#include <d3d11.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>
#include <wrl.h>

struct uint3;
struct uint2;

namespace WaveFront
{
	class DX11Wrapper
	{
	public:
		void Init();
		inline Microsoft::WRL::ComPtr<ID3D11Device> GetDevice() { return m_D3dDevice; };
		inline Microsoft::WRL::ComPtr<ID3D11DeviceContext> GetContext() { return m_D3dDeviceContext; };

		Microsoft::WRL::ComPtr<ID3D11Texture2D> CreateTexture2D(const uint3& a_ResDepth);
		Microsoft::WRL::ComPtr<ID3D11Texture2D> ResizeTexture2D(Microsoft::WRL::ComPtr<ID3D11Texture2D>& a_Tex, const uint2& a_NewSize);

		ID3D11Texture2D* m_D3D11PixelBufferCombined;


	private:
		Microsoft::WRL::ComPtr<ID3D11Device> m_D3dDevice = nullptr;
		Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_D3dDeviceContext = nullptr;


	};
}