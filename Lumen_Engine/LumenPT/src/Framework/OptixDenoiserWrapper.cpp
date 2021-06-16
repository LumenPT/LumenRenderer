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
    assert(a_InitParams.m_ServiceLocator != nullptr);
    assert(a_InitParams.m_ServiceLocator->m_OptixWrapper != nullptr);

    m_InitParams = a_InitParams;
    WaveFront::OptixWrapper& optixWrapper = *(m_InitParams.m_ServiceLocator->m_OptixWrapper);

    OptixDenoiserOptions options = {};
    options.guideAlbedo = 0;
    options.guideNormal = 0;
    
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

    TestInput.Resize(m_InitParams.m_InputWidth * m_InitParams.m_InputHeight * sizeof(float3));
    TestOutput.Resize(m_InitParams.m_InputWidth * m_InitParams.m_InputHeight * sizeof(float3));
}

void OptixDenoiserWrapper::Denoise(const OptixDenoiserDenoiseParams& a_DenoiseParams)
{
    assert(a_DenoiseParams.m_PostProcessLaunchParams != nullptr);

    OptixDenoiserParams optixDenoiserParams = {};
    optixDenoiserParams.denoiseAlpha = false;
    optixDenoiserParams.blendFactor = 0.0f;

    //MemoryBuffer test1(m_InitParams.m_InputWidth * m_InitParams.m_InputHeight * sizeof(float3));
    //MemoryBuffer test2(m_InitParams.m_InputWidth * m_InitParams.m_InputHeight * sizeof(float3));

    OptixDenoiserGuideLayer guideLayer = {};

    OptixDenoiserLayer inputOutputLayer = {};

    OptixImage2D& colorTex = inputOutputLayer.input;
    colorTex.data = /*static_cast<CUdeviceptr>(test1.GetCUDAPtr())*/a_DenoiseParams.m_ColorInput;
    colorTex.width = m_InitParams.m_InputWidth;
    colorTex.height = m_InitParams.m_InputHeight;
    colorTex.pixelStrideInBytes = sizeof(float3);
    colorTex.rowStrideInBytes = colorTex.pixelStrideInBytes * colorTex.width;
    colorTex.format = OPTIX_PIXEL_FORMAT_FLOAT3;

    OptixImage2D& outputTex = inputOutputLayer.output;
    outputTex.data = /*static_cast<CUdeviceptr>(test2.GetCUDAPtr())*/a_DenoiseParams.m_Output;
    outputTex.width = m_InitParams.m_InputWidth;
    outputTex.height = m_InitParams.m_InputHeight;
    outputTex.pixelStrideInBytes = sizeof(float3);
    colorTex.rowStrideInBytes = colorTex.pixelStrideInBytes * colorTex.width;
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

    //OptixDenoiser m_Denoiser;

    //CUdeviceptr m_state;
    //size_t m_state_size;
    //CUdeviceptr m_scratch;
    //size_t m_scratch_size;

    //size_t textureSize = sizeof(float) * 3 * 800 * 600;

    //CUdeviceptr TestInput;
    //CHECKCUDAERROR(cudaMalloc(
    //    reinterpret_cast<void**>(&TestInput),
    //    textureSize
    //));

    //CUdeviceptr TestOutput;
    //CHECKCUDAERROR(cudaMalloc(
    //    reinterpret_cast<void**>(&TestOutput),
    //    textureSize
    //));

    //OptixDenoiserOptions denoiserOptions = {};
    //denoiserOptions.guideAlbedo = 0;
    //denoiserOptions.guideNormal = 0;

    //OptixDenoiserModelKind denoiserModelKind = {};
    //denoiserModelKind = OPTIX_DENOISER_MODEL_KIND_LDR;

    //WaveFront::OptixWrapper& optixWrapper = *(m_InitParams.m_ServiceLocator->m_OptixWrapper);

    //CHECKOPTIXRESULT(optixDenoiserCreate(optixWrapper.GetDeviceContext(), denoiserModelKind, &denoiserOptions, &m_Denoiser));

    //OptixDenoiserSizes denoiserSizes = {};
    //CHECKOPTIXRESULT(optixDenoiserComputeMemoryResources(m_Denoiser, 800, 600, &denoiserSizes)); //TODO: might have to set max input to 4K resolution

    //m_state_size = denoiserSizes.stateSizeInBytes;
    //m_scratch_size = denoiserSizes.withoutOverlapScratchSizeInBytes;

    //CHECKCUDAERROR(cudaMalloc(
    //    reinterpret_cast<void**>(&m_state),
    //    m_state_size
    //));

    //CHECKCUDAERROR(cudaMalloc(
    //    reinterpret_cast<void**>(&m_scratch),
    //    m_scratch_size
    //));

    //CHECKOPTIXRESULT(optixDenoiserSetup(
    //    m_Denoiser,
    //    0,
    //    800,
    //    600,
    //    m_state,
    //    m_state_size,
    //    m_scratch,
    //    m_scratch_size
    //));

    //OptixDenoiserParams optixDenoiserParams = {};
    //optixDenoiserParams.denoiseAlpha = false;
    //optixDenoiserParams.blendFactor = 0.0f;

    //OptixDenoiserGuideLayer guideLayer = {};

    //OptixDenoiserLayer inputOutputLayer = {};

    //OptixImage2D& colorTex = inputOutputLayer.input;
    //colorTex.data = TestInput;
    //colorTex.width = 800;
    //colorTex.height = 600;
    //colorTex.pixelStrideInBytes = sizeof(float3);
    //colorTex.rowStrideInBytes = colorTex.pixelStrideInBytes * colorTex.width;
    //colorTex.format = OPTIX_PIXEL_FORMAT_FLOAT3;

    //OptixImage2D& outputTex = inputOutputLayer.output;
    //outputTex.data = TestOutput;
    //outputTex.width = 800;
    //outputTex.height = 600;
    //outputTex.pixelStrideInBytes = sizeof(float3);
    //colorTex.rowStrideInBytes = colorTex.pixelStrideInBytes * colorTex.width;
    //outputTex.format = OPTIX_PIXEL_FORMAT_FLOAT3;

    //CHECKLASTCUDAERROR;
    //auto result = optixDenoiserInvoke(
    //    m_Denoiser,
    //    0,
    //    &optixDenoiserParams,
    //    m_state,
    //    m_state_size,
    //    &guideLayer,
    //    &inputOutputLayer,
    //    1,
    //    0,
    //    0,
    //    m_scratch,
    //    m_scratch_size
    //);

    //CHECKOPTIXRESULT(result);
    //CHECKLASTCUDAERROR;

}
