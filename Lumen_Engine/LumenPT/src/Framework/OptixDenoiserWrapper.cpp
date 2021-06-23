#include "OptixDenoiserWrapper.h"

#include <assert.h>

#include "PTServiceLocator.h"
#include "OptixWrapper.h"
#include "CudaUtilities.h"
#include "../Tools/SnapShotProcessing.cuh"

OptixDenoiserWrapper::~OptixDenoiserWrapper()
{
    optixDenoiserDestroy(m_Denoiser);
}

void OptixDenoiserWrapper::Initialize(const OptixDenoiserInitParams& a_InitParams)
{
    assert(a_InitParams.m_ServiceLocator != nullptr);
    assert(a_InitParams.m_ServiceLocator->m_OptixWrapper != nullptr);

    m_InitParams = a_InitParams;
    WaveFront::OptixWrapper& optixWrapper = *(m_InitParams.m_ServiceLocator->m_OptixWrapper);

    OptixDenoiserOptions options = {};
    options.guideAlbedo = 1;
    options.guideNormal = 1;
    
    OptixDenoiserModelKind denoiserModelKind = {};
    denoiserModelKind = OPTIX_DENOISER_MODEL_KIND_LDR;

    CHECKOPTIXRESULT(optixDenoiserCreate(optixWrapper.GetDeviceContext(), denoiserModelKind, &options, &m_Denoiser));

    OptixDenoiserSizes denoiserSizes = {};
    CHECKOPTIXRESULT(optixDenoiserComputeMemoryResources(m_Denoiser, m_InitParams.m_InputWidth, m_InitParams.m_InputHeight, &denoiserSizes)); //TODO: might have to set max input to 4K resolution

    m_state.Resize(denoiserSizes.stateSizeInBytes);
    m_scratch.Resize(denoiserSizes.withoutOverlapScratchSizeInBytes);

    CHECKOPTIXRESULT(optixDenoiserSetup(
        m_Denoiser,
        0,
        m_InitParams.m_InputWidth,
        m_InitParams.m_InputHeight,
        static_cast<CUdeviceptr>(m_state.GetCUDAPtr()),
        m_state.GetSize(),
        static_cast<CUdeviceptr>(m_scratch.GetCUDAPtr()),
        m_scratch.GetSize()
    ));

    ColorInput.Resize(m_InitParams.m_InputWidth * m_InitParams.m_InputHeight * sizeof(float3));
    AlbedoInput.Resize(m_InitParams.m_InputWidth * m_InitParams.m_InputHeight * sizeof(float3));
    NormalInput.Resize(m_InitParams.m_InputWidth * m_InitParams.m_InputHeight * sizeof(float3));
    ColorOutput.Resize(m_InitParams.m_InputWidth * m_InitParams.m_InputHeight * sizeof(float3));

    m_OptixDenoiserInputTex.m_Memory = std::make_unique<CudaGLTexture>(GL_RGB32F, m_InitParams.m_InputWidth,
        m_InitParams.m_InputHeight, 3 * sizeof(float));;
    m_OptixDenoiserAlbedoInputTex.m_Memory = std::make_unique<CudaGLTexture>(GL_RGB32F, m_InitParams.m_InputWidth,
        m_InitParams.m_InputHeight, 3 * sizeof(float));;
    m_OptixDenoiserNormalInputTex.m_Memory = std::make_unique<CudaGLTexture>(GL_RGB32F, m_InitParams.m_InputWidth,
        m_InitParams.m_InputHeight, 3 * sizeof(float));;
    m_OptixDenoiserOutputTex.m_Memory = std::make_unique<CudaGLTexture>(GL_RGB32F, m_InitParams.m_InputWidth,
        m_InitParams.m_InputHeight, 3 * sizeof(float));;
}

void OptixDenoiserWrapper::Denoise(const OptixDenoiserDenoiseParams& a_DenoiseParams)
{
    assert(a_DenoiseParams.m_PostProcessLaunchParams != nullptr);

    OptixDenoiserParams optixDenoiserParams = {};
    optixDenoiserParams.denoiseAlpha = false;
    optixDenoiserParams.blendFactor = 0.0f;

    OptixDenoiserGuideLayer guideLayer = {};

    OptixImage2D& albedoTex = guideLayer.albedo;
    albedoTex.data = a_DenoiseParams.m_AlbedoInput;
    albedoTex.width = m_InitParams.m_InputWidth;
    albedoTex.height = m_InitParams.m_InputHeight;
    albedoTex.pixelStrideInBytes = sizeof(float3);
    albedoTex.rowStrideInBytes = albedoTex.pixelStrideInBytes * albedoTex.width;
    albedoTex.format = OPTIX_PIXEL_FORMAT_FLOAT3;

    OptixImage2D& normalTex = guideLayer.normal;
    normalTex.data = a_DenoiseParams.m_NormalInput;
    normalTex.width = m_InitParams.m_InputWidth;
    normalTex.height = m_InitParams.m_InputHeight;
    normalTex.pixelStrideInBytes = sizeof(float3);
    normalTex.rowStrideInBytes = normalTex.pixelStrideInBytes * normalTex.width;
    normalTex.format = OPTIX_PIXEL_FORMAT_FLOAT3;

    OptixDenoiserLayer inputOutputLayer = {};

    OptixImage2D& colorTex = inputOutputLayer.input;
    colorTex.data = a_DenoiseParams.m_ColorInput;
    colorTex.width = m_InitParams.m_InputWidth;
    colorTex.height = m_InitParams.m_InputHeight;
    colorTex.pixelStrideInBytes = sizeof(float3);
    colorTex.rowStrideInBytes = colorTex.pixelStrideInBytes * colorTex.width;
    colorTex.format = OPTIX_PIXEL_FORMAT_FLOAT3;

    OptixImage2D& outputTex = inputOutputLayer.output;
    outputTex.data = a_DenoiseParams.m_Output;
    outputTex.width = m_InitParams.m_InputWidth;
    outputTex.height = m_InitParams.m_InputHeight;
    outputTex.pixelStrideInBytes = sizeof(float3);
    outputTex.rowStrideInBytes = outputTex.pixelStrideInBytes * outputTex.width;
    outputTex.format = OPTIX_PIXEL_FORMAT_FLOAT3;

    CHECKLASTCUDAERROR;
    auto result = optixDenoiserInvoke(
        m_Denoiser,
        0,
        &optixDenoiserParams,
        m_state.GetCUDAPtr(),
        m_state.GetSize(),
        &guideLayer,
        &inputOutputLayer,
        1,
        0,
        0,
        m_scratch.GetCUDAPtr(),
        m_scratch.GetSize()
        );

    CHECKOPTIXRESULT(result);
    CHECKLASTCUDAERROR;
}

void OptixDenoiserWrapper::UpdateDebugTextures()
{
    SeparateOptixDenoiserBufferCPU(
        m_InitParams.m_InputWidth * m_InitParams.m_InputHeight,
        ColorInput.GetDevicePtr<float3>(),
        AlbedoInput.GetDevicePtr<float3>(),
        NormalInput.GetDevicePtr<float3>(),
        ColorOutput.GetDevicePtr<float3>(),
        m_OptixDenoiserInputTex.m_Memory->GetDevicePtr<float3>(),
        m_OptixDenoiserAlbedoInputTex.m_Memory->GetDevicePtr<float3>(),
        m_OptixDenoiserNormalInputTex.m_Memory->GetDevicePtr<float3>(),
        m_OptixDenoiserOutputTex.m_Memory->GetDevicePtr<float3>()
    );
}
