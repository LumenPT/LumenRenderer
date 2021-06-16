#include "OptixDenoiserWrapper.h"

#include <assert.h>

#include "PTServiceLocator.h"
#include "OptixWrapper.h"
#include "CudaUtilities.h"

OptixDenoiserWrapper::~OptixDenoiserWrapper()
{
    optixDenoiserDestroy(m_Denoiser);
}

void OptixDenoiserWrapper::Initialize(const OptixDenoiserInitParams& a_InitParams)
{
    //assert(a_InitParams.m_ServiceLocator != nullptr);
    //assert(a_InitParams.m_ServiceLocator->m_OptixWrapper != nullptr);

    //m_InitParams = a_InitParams;
    //WaveFront::OptixWrapper& optixWrapper = *(m_InitParams.m_ServiceLocator->m_OptixWrapper);

    //OptixDenoiserOptions options = {}; //this struct is missing some options?
    //options.inputKind = OPTIX_DENOISER_INPUT_RGB; //TODO: add albedo and normal at later date
    //
    //CHECKOPTIXRESULT(optixDenoiserCreate(optixWrapper.GetDeviceContext(), &options, &m_Denoiser));

    //OptixDenoiserModelKind denoiserModelKind = {};
    //denoiserModelKind = OPTIX_DENOISER_MODEL_KIND_LDR;

    //CHECKOPTIXRESULT(optixDenoiserSetModel(m_Denoiser, denoiserModelKind, nullptr, 0));

    //OptixDenoiserSizes denoiserSizes = {};
    //CHECKOPTIXRESULT(optixDenoiserComputeMemoryResources(m_Denoiser, m_InitParams.m_InputWidth, m_InitParams.m_InputHeight, &denoiserSizes)); //TODO: might have to set max input to 4K resolution

    //m_state.Resize(denoiserSizes.stateSizeInBytes);
    //m_scratch.Resize(denoiserSizes.withoutOverlapScratchSizeInBytes);

    //CHECKOPTIXRESULT(optixDenoiserSetup(
    //    m_Denoiser,
    //    0,
    //    m_InitParams.m_InputWidth,
    //    m_InitParams.m_InputHeight,
    //    static_cast<CUdeviceptr>(m_state.GetCUDAPtr()),
    //    m_state.GetSize(),
    //    static_cast<CUdeviceptr>(m_scratch.GetCUDAPtr()),
    //    m_scratch.GetSize()
    //));

    //TestInput.Resize(m_InitParams.m_InputWidth * m_InitParams.m_InputHeight * sizeof(float3));
    //TestOutput.Resize(m_InitParams.m_InputWidth * m_InitParams.m_InputHeight * sizeof(float3));
}

void OptixDenoiserWrapper::Denoise(const OptixDenoiserDenoiseParams& a_DenoiseParams)
{
    //assert(a_DenoiseParams.m_PostProcessLaunchParams != nullptr);

    //OptixDenoiserParams optixDenoiserParams = {};
    //optixDenoiserParams.denoiseAlpha = false;
    //optixDenoiserParams.blendFactor = 0.0f;

    ////MemoryBuffer test1(m_InitParams.m_InputWidth * m_InitParams.m_InputHeight * sizeof(float3));
    ////MemoryBuffer test2(m_InitParams.m_InputWidth * m_InitParams.m_InputHeight * sizeof(float3));

    //std::vector<OptixImage2D> inputLayers;
    //OptixImage2D& colorTex = inputLayers.emplace_back();
    //colorTex.data = /*static_cast<CUdeviceptr>(test1.GetCUDAPtr())*/a_DenoiseParams.m_ColorInput;
    //colorTex.width = m_InitParams.m_InputWidth;
    //colorTex.height = m_InitParams.m_InputHeight;
    //colorTex.pixelStrideInBytes = sizeof(float3);
    //colorTex.rowStrideInBytes = colorTex.pixelStrideInBytes * colorTex.width;
    //colorTex.format = OPTIX_PIXEL_FORMAT_FLOAT3;

    //OptixImage2D outputTex;
    //outputTex.data = /*static_cast<CUdeviceptr>(test2.GetCUDAPtr())*/a_DenoiseParams.m_Output;
    //outputTex.width = m_InitParams.m_InputWidth;
    //outputTex.height = m_InitParams.m_InputHeight;
    //outputTex.pixelStrideInBytes = sizeof(float3);
    //colorTex.rowStrideInBytes = colorTex.pixelStrideInBytes * colorTex.width;
    //outputTex.format = OPTIX_PIXEL_FORMAT_FLOAT3;

    /*CHECKLASTCUDAERROR;
    auto result = optixDenoiserInvoke(
        m_Denoiser,
        0,
        &optixDenoiserParams,
        m_state.GetCUDAPtr(),
        m_state.GetSize(),
        inputLayers.data(),
        1,
        0,
        0,
        &outputTex,
        m_scratch.GetCUDAPtr(),
        m_scratch.GetSize()
        );

    CHECKOPTIXRESULT(result);
    CHECKLASTCUDAERROR;*/

}
