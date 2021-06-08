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

    OptixDenoiserOptions options = {}; //this struct is missing some options?
    options.inputKind = OPTIX_DENOISER_INPUT_RGB; //TODO: add albedo and normal at later date
    
    CHECKOPTIXRESULT(optixDenoiserCreate(optixWrapper.GetDeviceContext(), &options, &m_Denoiser));

    OptixDenoiserModelKind denoiserModelKind = {};
    denoiserModelKind = OPTIX_DENOISER_MODEL_KIND_LDR;

    CHECKOPTIXRESULT(optixDenoiserSetModel(m_Denoiser, denoiserModelKind, nullptr, 0));

    OptixDenoiserSizes denoiserSizes = {};
    CHECKOPTIXRESULT(optixDenoiserComputeMemoryResources(m_Denoiser, m_InitParams.m_InputWidth, m_InitParams.m_InputHeight, &denoiserSizes)); //TODO: might have to set max input to 4K resolution

    optixDenoiserSetup(
        m_Denoiser,
        nullptr,
        m_InitParams.m_InputWidth,
        m_InitParams.m_InputHeight,
        m_State,
        m_StateSize,
        m_Scratch,
        m_ScratchSize
        );
}

void OptixDenoiserWrapper::Denoise(const OptixDenoiserDenoiseParams& a_DenoiseParams)
{
    OptixDenoiserParams optixDenoiserParams = {};
    optixDenoiserParams.denoiseAlpha = false;
    optixDenoiserParams.blendFactor = 0.0f;

    auto result = optixDenoiserInvoke(
        m_Denoiser,
        0,
        &optixDenoiserParams,
        m_State,
        m_StateSize,
        nullptr, //TODO
        1, //TODO
        0,
        0,
        nullptr, //TODO
        m_Scratch,
        m_ScratchSize
        );

    CHECKOPTIXRESULT(result);
    //TODO: check result
}
