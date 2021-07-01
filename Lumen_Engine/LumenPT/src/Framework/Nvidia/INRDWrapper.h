#pragma once

//#include <wrl.h> //TODO: add this back
//#include <d3d11.h>
//#include "gltf.h"

class PTServiceLocator;
class ID3D11Texture2D;
class Camera;

struct NRDWrapperInitParams
{
	int m_InputImageWidth = -1;
	int m_InputImageHeight = -1;
	PTServiceLocator* m_pServiceLocator = nullptr;

	ID3D11Texture2D* m_InputDiffTex;
	ID3D11Texture2D* m_InputSpecTex;
	ID3D11Texture2D* m_NormalRoughnessTex;
	ID3D11Texture2D* m_MotionVectorTex;
	ID3D11Texture2D* m_ViewZTex;
	ID3D11Texture2D* m_OutputDiffTex;
	ID3D11Texture2D* m_OutputSpecTex;
};

struct NRDWrapperEvaluateParams
{
	int m_InputImageWidth = -1;
	int m_InputImageHeight = -1;
	Camera* m_Camera;
	/*ID3D11Resource* m_InputDiffTex;
	ID3D11Resource* m_InputSpecTex;
	ID3D11Resource* m_NormalRoughnessTex;;
	ID3D11Resource* m_MotionVectorTex;
	ID3D11Resource* m_ViewZTex;
	ID3D11Resource* m_OutputTex;*/
};

class INRDWrapper
{
public:

	INRDWrapper() = default;
	virtual ~INRDWrapper() = default;

	virtual void Initialize(NRDWrapperInitParams& a_InitParams) = 0;

	virtual void Denoise(NRDWrapperEvaluateParams& a_DenoiseParams) = 0;

protected:

};