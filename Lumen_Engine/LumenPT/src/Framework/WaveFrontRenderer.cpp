#if defined(WAVEFRONT)
#include "WaveFrontRenderer.h"
#include "PTMesh.h"
#include "PTScene.h"
#include "PTPrimitive.h"
#include "PTVolume.h"
#include "AccelerationStructure.h"
#include "Material.h"
#include "Texture.h"
#include "MemoryBuffer.h"
#include "OutputBuffer.h"
#include "ShaderBindingTableGen.h"
#include "CudaUtilities.h"
#include "../Shaders/CppCommon/LumenPTConsts.h"
#include "../CUDAKernels/WaveFrontKernels.cuh"
#include "../CUDAKernels/WaveFrontKernels/GPUShadingKernels.cuh"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <filesystem>
#include "Cuda/cuda.h"
#include "Cuda/cuda_runtime.h"
#include "Optix/optix_stubs.h"
#include <glm/gtx/compatibility.hpp>

WaveFrontRenderer::WaveFrontRenderer(const InitializationData& a_InitializationData)
    :
m_ServiceLocator({}),
m_DeviceContext(nullptr),
m_PipelineRays(nullptr),
m_PipelineShadowRays(nullptr),
m_PipelineRaysLaunchParams(nullptr),
m_PipelineShadowRaysLaunchParams(nullptr),
m_RaysSBTGenerator(nullptr),
m_ShadowRaysSBTGenerator(nullptr),
m_ProgramGroups({}),
m_OutputBuffer(nullptr),
m_SBTBuffer(nullptr),
m_RayBatchIndices({0}),
m_HitBufferIndices({0}),
m_ResultBuffer(nullptr),
m_PixelBufferMultiChannel(nullptr),
m_PixelBufferSingleChannel(nullptr),
m_IntersectionRayBatches(),
m_IntersectionBuffers(),
m_ShadowRayBatch(nullptr),
m_LightBufferTemp(nullptr),
m_Texture(nullptr),
m_RenderResolution(max(a_InitializationData.m_RenderResolution, s_minResolution)),
m_OutputResolution(max(a_InitializationData.m_OutputResolution, s_minResolution)),
m_MaxDepth(max(a_InitializationData.m_MaxDepth, s_minDepth)),
m_RaysPerPixel(max(a_InitializationData.m_RaysPerPixel, s_minRaysPerPixel)),
m_ShadowRaysPerPixel(max(a_InitializationData.m_ShadowRaysPerPixel, s_minShadowRaysPerPixel)),
m_FrameCount(0),
m_Initialized(false)
{

    m_Initialized = Initialize(a_InitializationData);
    if(!m_Initialized)
    {
        std::fprintf(stderr, "Initialization of wavefront renderer unsuccessful");
        abort();
    }

    m_RaysSBTGenerator = std::make_unique<ShaderBindingTableGenerator>();
    m_ShadowRaysSBTGenerator = std::make_unique<ShaderBindingTableGenerator>();

    m_Texture = std::make_unique<Texture>(LumenPTConsts::gs_AssetDirectory + "debugTex.jpg");

    m_ServiceLocator.m_SBTGenerator = m_RaysSBTGenerator.get();
    m_ServiceLocator.m_Renderer = this;

    CreateShaderBindingTables();
	
}

WaveFrontRenderer::~WaveFrontRenderer()
{}

bool WaveFrontRenderer::Initialize(const InitializationData& a_InitializationData)
{
    bool success = true;
    //Temporary(put into init data)
    const std::string shaderPath = LumenPTConsts::gs_ShaderPathBase + "WaveFrontShaders.ptx";

    InitializeContext();
    CreatePipelineBuffers();
    CreateOutputBuffer();
    CreateDataBuffers();
    SetupInitialBufferIndices();

    cudaDeviceSynchronize();

    CHECKLASTCUDAERROR;

    return success;

}

void WaveFrontRenderer::InitializeContext()
{

    cudaFree(0);
    CUcontext cu_ctx = 0;
    CHECKOPTIXRESULT(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &WaveFrontRenderer::DebugCallback;
    options.logCallbackLevel = 4;
    CHECKOPTIXRESULT(optixDeviceContextCreate(cu_ctx, &options, &m_DeviceContext));

}

OptixPipelineCompileOptions WaveFrontRenderer::CreatePipelineOptions(
    const std::string& a_LaunchParamName,
    unsigned int a_NumPayloadValues, 
    unsigned int a_NumAttributes) const
{

    OptixPipelineCompileOptions pipelineOptions = {};
    pipelineOptions.usesMotionBlur = false;
    pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipelineOptions.numPayloadValues = std::clamp(a_NumPayloadValues, 0u, 8u);
    pipelineOptions.numAttributeValues = std::clamp(a_NumAttributes, 2u, 8u);
    pipelineOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG;
    pipelineOptions.pipelineLaunchParamsVariableName = a_LaunchParamName.c_str();
    pipelineOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE & OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

    return pipelineOptions;

}



bool WaveFrontRenderer::CreatePipelines(const std::string& a_ShaderPath)
{

    bool success = true;

    const std::string resolveRaysParams = "launchParams";
    const std::string resolveRaysRayGenFuncName = "__raygen__UberGenShader";
    //const std::string resolveRaysRayGenFuncName = "__raygen__ResolveRaysRayGen";
    const std::string resolveRaysHitFuncName = "__closesthit__UberClosestHit";
    const std::string resolveRaysMissFuncName = "__miss__UberMiss";
    const std::string resolveRaysAnyhitFuncName = "__miss__UberAnyHit";

    const std::string resolveShadowRaysParams = "launchParams";
    //const std::string resolveShadowRaysRayGenFuncName = "__raygen__ResolveShadowRaysRayGen";
    //const std::string resolveShadowRaysHitFuncName = "__anyhit__ResolveShadowRaysAnyHit";
    //const std::string resolveShadowRaysMissFuncName = "__miss__ResolveShadowRaysMiss";

    OptixPipelineCompileOptions compileOptions = CreatePipelineOptions(resolveRaysParams, 2, 2);

    OptixModule shaderModule = CreateModule(a_ShaderPath, compileOptions);
    if (shaderModule == nullptr) { return false; }

    success &= CreatePipeline(
        shaderModule,
        compileOptions,
        PipelineType::RESOLVE_RAYS, 
        resolveRaysRayGenFuncName, 
        resolveRaysHitFuncName,
        resolveRaysMissFuncName,
        m_PipelineRays);

    //optixModuleDestroy(shaderModule);

    compileOptions = CreatePipelineOptions(resolveShadowRaysParams, 2, 2);
    shaderModule = CreateModule(a_ShaderPath, compileOptions);
    if (shaderModule == nullptr) { return false; }

    //success &= CreatePipeline(
    //    shaderModule,
    //    compileOptions,
    //    PipelineType::RESOLVE_SHADOW_RAYS, 
    //    resolveShadowRaysRayGenFuncName, 
    //    resolveShadowRaysHitFuncName,
    //    resolveShadowRaysMissFuncName,
    //    m_PipelineShadowRays);

    //optixModuleDestroy(shaderModule);

    return success;

}

//TODO: implement necessary Optix shaders for wavefront algorithm
//Shaders:
// - RayGen: trace rays from ray definitions in ray batch, different methods for radiance or shadow.
// - Miss: is this shader necessary, miss result by default in intersection buffer?
// - ClosestHit: radiance trace, report intersection into intersection buffer.
// - AnyHit: shadow trace, report if any hit in between certain max distance.
bool WaveFrontRenderer::CreatePipeline(
    const OptixModule& a_Module,
    const OptixPipelineCompileOptions& a_PipelineOptions,
    PipelineType a_Type, 
    const std::string& a_RayGenFuncName, 
    const std::string& a_HitFuncName,
    const std::string& a_MissFuncName,
    OptixPipeline& a_Pipeline)
{

    OptixProgramGroup rayGenProgram = nullptr;
    OptixProgramGroup hitProgram = nullptr;
    OptixProgramGroup missProgram = nullptr;

    OptixProgramGroupDesc rgGroupDesc = {};
    rgGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rgGroupDesc.raygen.entryFunctionName = a_RayGenFuncName.c_str();
    rgGroupDesc.raygen.module = a_Module;

    OptixProgramGroupDesc htGroupDesc = {};
    htGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

    OptixProgramGroupDesc msGroupDesc = {};
    msGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;

    switch (a_Type)
    {

        case PipelineType::RESOLVE_RAYS:
            {
                htGroupDesc.hitgroup.entryFunctionNameCH = a_HitFuncName.c_str();
                htGroupDesc.hitgroup.moduleCH = a_Module;








void WaveFrontRenderer::CreateOutputBuffer()
{

    m_OutputBuffer = std::make_unique<::OutputBuffer>(m_OutputResolution.x, m_OutputResolution.y);

}

void WaveFrontRenderer::CreateDataBuffers()
{

    const unsigned numPixels = m_RenderResolution.x * m_RenderResolution.y;
    const unsigned numOutputChannels = ResultBuffer::s_NumOutputChannels;

    //const unsigned int lightBuffer = LightBuffer::

    const unsigned pixelBufferEmptySize = sizeof(PixelBuffer);
    const unsigned pixelDataStructSize = sizeof(float3);

    //Allocate pixel buffer.
    m_PixelBufferMultiChannel = std::make_unique<MemoryBuffer>(
        static_cast<size_t>(pixelBufferEmptySize) + 
        static_cast<size_t>(numPixels) *
        static_cast<size_t>(numOutputChannels) *
        static_cast<size_t>(pixelDataStructSize));
    m_PixelBufferMultiChannel->Write(numPixels, 0);
    m_PixelBufferMultiChannel->Write(numOutputChannels, sizeof(PixelBuffer::m_NumPixels));

    m_PixelBufferSingleChannel = std::make_unique<MemoryBuffer>(
        static_cast<size_t>(pixelBufferEmptySize) + 
        static_cast<size_t>(numPixels)*
        static_cast<size_t>(pixelDataStructSize));
    m_PixelBufferSingleChannel->Write(numPixels, 0);
    m_PixelBufferSingleChannel->Write(1, sizeof(PixelBuffer::m_NumPixels));


    const PixelBuffer* pixelBufferPtr = m_PixelBufferMultiChannel->GetDevicePtr<PixelBuffer>();

    //Allocate result buffer.
    m_ResultBuffer = std::make_unique<MemoryBuffer>(sizeof(ResultBuffer));
    m_ResultBuffer->Write(&pixelBufferPtr, sizeof(ResultBuffer::m_PixelBuffer), 0);

    const unsigned rayBatchEmptySize = sizeof(IntersectionRayBatch);
    const unsigned rayDataStructSize = sizeof(IntersectionRayData);

    //Allocate and initialize ray batches.
    int batchIndex = 0;
    for(auto& rayBatch : m_IntersectionRayBatches)
    {
        rayBatch = std::make_unique<MemoryBuffer>(
            static_cast<size_t>(rayBatchEmptySize) + 
            static_cast<size_t>(numPixels) * 
            static_cast<size_t>(m_RaysPerPixel) * 
            static_cast<size_t>(rayDataStructSize));
        //rayBatch->Write(numPixels, 0);
        //rayBatch->Write(m_RaysPerPixel, sizeof(RayBatch::m_NumPixels));

        ResetIntersectionRayBatch(rayBatch->GetDevicePtr<IntersectionRayBatch>(), numPixels, m_RaysPerPixel);

    }

    cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;

    const unsigned intersectionBufferEmptySize = sizeof(IntersectionBuffer);
    const unsigned intersectionDataStructSize = sizeof(IntersectionData);

    unsigned bufferIndex = 0;
    for(auto& intersectionBuffer : m_IntersectionBuffers)
    {
        intersectionBuffer = std::make_unique<MemoryBuffer>(
           static_cast<size_t>(intersectionBufferEmptySize) +
           static_cast<size_t>(numPixels) * 
           static_cast<size_t>(m_RaysPerPixel) *
           static_cast<size_t>(intersectionDataStructSize));
        intersectionBuffer->Write(numPixels, 0);
        intersectionBuffer->Write(m_RaysPerPixel, sizeof(IntersectionBuffer::m_NumPixels));
    }

    const unsigned ShadowRayBatchEmptySize = sizeof(ShadowRayBatch);
    const unsigned ShadowRayDataStructSize = sizeof(ShadowRayData);

    m_ShadowRayBatch = std::make_unique<MemoryBuffer>(
        static_cast<size_t>(ShadowRayBatchEmptySize) + 
        static_cast<size_t>(m_MaxDepth) * 
        static_cast<size_t>(numPixels) * 
        static_cast<size_t>(m_ShadowRaysPerPixel) * 
        static_cast<size_t>(ShadowRayDataStructSize));
    //m_ShadowRayBatch->Write(m_MaxDepth, 0);
    //m_ShadowRayBatch->Write(numPixels, sizeof(ShadowRayBatch::m_MaxDepth));
    //m_ShadowRayBatch->Write(m_ShadowRaysPerPixel, sizeof(ShadowRayBatch::m_MaxDepth) + sizeof(ShadowRayBatch::m_NumPixels));

    ResetShadowRayBatch(m_ShadowRayBatch->GetDevicePtr<ShadowRayBatch>(), m_MaxDepth, numPixels, m_ShadowRaysPerPixel);

    cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;



    const unsigned LightBufferEmptySize = sizeof(LightDataBuffer);
    const unsigned LightDataStructSize = sizeof(TriangleLight);

    m_LightBufferTemp = std::make_unique<MemoryBuffer>(
        static_cast<size_t>(LightBufferEmptySize) +
        static_cast<size_t>(3) *
        static_cast<size_t>(LightDataStructSize));


    TriangleLight lights[3] = {
        {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}},
        {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}},
        {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}} };

    LightDataBuffer* tempLightBuffer = reinterpret_cast<LightDataBuffer*>(malloc(sizeof(LightDataBuffer) + sizeof(lights)));
    tempLightBuffer->m_Lights[0] = lights[0];
    tempLightBuffer->m_Lights[1] = lights[1];
    tempLightBuffer->m_Lights[2] = lights[2];

    m_LightBufferTemp->Write(
        tempLightBuffer, 
        static_cast<size_t>(LightBufferEmptySize) +
        static_cast<size_t>(3) *
        static_cast<size_t>(LightDataStructSize));



    

}

void WaveFrontRenderer::SetupInitialBufferIndices()
{

    m_RayBatchIndices[static_cast<unsigned>(RayBatchTypeIndex::PRIM_RAYS_PREV_FRAME)] = 0;
    m_RayBatchIndices[static_cast<unsigned>(RayBatchTypeIndex::CURRENT_RAYS)] = s_NumRayBatchTypes - 1;
    m_RayBatchIndices[static_cast<unsigned>(RayBatchTypeIndex::SECONDARY_RAYS)] = 1;

    m_HitBufferIndices[static_cast<unsigned>(HitBufferTypeIndex::PRIM_HITS_PREV_FRAME)] = 0;
    m_HitBufferIndices[static_cast<unsigned>(HitBufferTypeIndex::CURRENT_HITS)] = s_NumHitBufferTypes - 1;
}







GLuint WaveFrontRenderer::TraceFrame()
{

    CHECKLASTCUDAERROR;

    //Clear Pixel buffer
    PixelBuffer* pixelBufferMultiChannelDevPtr = m_PixelBufferMultiChannel->GetDevicePtr<PixelBuffer>();
    const unsigned numPixels = m_RenderResolution.x * m_RenderResolution.y;
    const unsigned channelsPerPixel = static_cast<unsigned>(ResultBuffer::s_NumOutputChannels);
    ResetPixelBuffer(pixelBufferMultiChannelDevPtr, numPixels, channelsPerPixel);

    //Generate Camera rays using CUDA kernel.
    float3 eye, u, v, w;
    m_Camera.SetAspectRatio(static_cast<float>(m_RenderResolution.x) / static_cast<float>(m_RenderResolution.y));
    m_Camera.GetVectorData(eye, u, v, w);
    const WaveFront::PrimRayGenLaunchParameters::DeviceCameraData cameraData(eye, u, v, w);

    //Get new Ray Batch to fill with Primary Rays (Either first or last ray batch, opposite of current PrimRaysPrevFrame batch)
    //Get a new array of indices to temporarily update the current array of indices.
    std::array<unsigned, s_NumRayBatchTypes> batchIndices{};
    GetRayBatchIndices(0, m_RayBatchIndices, batchIndices);

    //Get index to use to get the ray batch to use for the current ray buffer.
    const unsigned currentRayBatchIndex = batchIndices[static_cast<unsigned>(RayBatchTypeIndex::CURRENT_RAYS)];
    MemoryBuffer& currentRaysBatch = *m_IntersectionRayBatches[currentRayBatchIndex];

    //Generate primary rays using the setup parameters
    const PrimRayGenLaunchParameters primaryRayGenParams(m_RenderResolution, cameraData, currentRaysBatch.GetDevicePtr<IntersectionRayBatch>(), m_FrameCount);
    GeneratePrimaryRays(primaryRayGenParams);

    

    /*void* primRayBatchCuPtr = m_RayBatches[currentRayBatchIndex]->GetDevicePtr();
    SaveRayBatchToBMP(
        primRayBatchCuPtr, 
        m_RenderResolution.x, 
        m_RenderResolution.y, 
        m_RaysPerPixel, 
        frameSaveFilePath + "RayBatches/", 
        "PrimaryRays");*/

    OptixShaderBindingTable raysSBT = m_RaysSBTGenerator->GetTableDesc();

    //Initialize resolveRaysLaunchParameters with common variables between different waves.
    OptixLaunchParameters optixRaysLaunchParams{};


    /// <summary> /////////////////////////////
    /// IMPORTANT THIS GETS RAN AFTER SHADER BINDING TABLE GETS GENERATED!!
    /// </summary> ////////////////////////////
    optixRaysLaunchParams.m_TraversableHandle = dynamic_cast<PTScene&>(*m_Scene).GetSceneAccelerationStructure();

    uint3 resolutionAndDepth = make_uint3(m_RenderResolution.x, m_RenderResolution.y, 0);
    
    cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;

    //Loop
    //Trace buffer of rays using Optix ResolveRays pipeline
    //Calculate shading for intersections in buffer using CUDA kernel.
    for(unsigned waveIndex = 0; waveIndex < m_MaxDepth; ++waveIndex)
    {

        GetRayBatchIndices(waveIndex, m_RayBatchIndices, m_RayBatchIndices);
        GetHitBufferIndices(waveIndex, m_HitBufferIndices, m_HitBufferIndices);

        MemoryBuffer& primRaysPrevFrame =   *m_IntersectionRayBatches[m_RayBatchIndices[static_cast<unsigned>(RayBatchTypeIndex::PRIM_RAYS_PREV_FRAME)]];
        MemoryBuffer& currentRays =         *m_IntersectionRayBatches[m_RayBatchIndices[static_cast<unsigned>(RayBatchTypeIndex::CURRENT_RAYS)]];
        MemoryBuffer& secondaryRays =       *m_IntersectionRayBatches[m_RayBatchIndices[static_cast<unsigned>(RayBatchTypeIndex::SECONDARY_RAYS)]];

        MemoryBuffer& primHitsPrevFrame =   *m_IntersectionBuffers[m_HitBufferIndices[static_cast<unsigned>(HitBufferTypeIndex::PRIM_HITS_PREV_FRAME)]];
        MemoryBuffer& currentHits =         *m_IntersectionBuffers[m_HitBufferIndices[static_cast<unsigned>(HitBufferTypeIndex::CURRENT_HITS)]];

        void* primRayPrevFrameBatchCuPtr = primRaysPrevFrame.GetDevicePtr();
        void* currentRayBatchCuPtr = currentRays.GetDevicePtr();
        void* secondaryRayBatchCuPtr = secondaryRays.GetDevicePtr();

        //Resolution and current depth(, current depth = current wave index)
        resolutionAndDepth.z = waveIndex;

        optixRaysLaunchParams.m_ResolutionAndDepth = resolutionAndDepth;
        optixRaysLaunchParams.m_IntersectionRayBatch = currentRays.GetDevicePtr<IntersectionRayBatch>();
        optixRaysLaunchParams.m_IntersectionBuffer = currentHits.GetDevicePtr<IntersectionBuffer>();

        m_PipelineRaysLaunchParams->Write(optixRaysLaunchParams);

        cudaDeviceSynchronize();
        CHECKLASTCUDAERROR;

        //Launch OptiX ResolveRays pipeline to resolve all of the rays in Current Rays Batch (Secondary ray batch from previous wave).
        CHECKOPTIXRESULT(optixLaunch(
            m_PipelineRays,
            0,
            **m_PipelineRaysLaunchParams,
            m_PipelineRaysLaunchParams->GetSize(),
            &raysSBT,
            m_RenderResolution.x,
            m_RenderResolution.y,
            m_RaysPerPixel)); //Number of rays per pixel, number of samples per pixel.

        cudaDeviceSynchronize();
        CHECKLASTCUDAERROR;
        
        //cudaStreamSynchronize(0);
        /*cudaDeviceSynchronize();
        CHECKLASTCUDAERROR;*/

        /*void* hitBufferCuPtr = currentHits.GetDevicePtr();

        SaveIntersectionBufferToBMP(
            hitBufferCuPtr,
            m_RenderResolution.x,
            m_RenderResolution.y,
            m_RaysPerPixel,
            frameWaveSaveFilePath + "HitBuffers/",
            "currentHits");*/

        ShadingLaunchParameters shadingLaunchParams(
            resolutionAndDepth, 
            primRaysPrevFrame.GetDevicePtr<IntersectionRayBatch>(),
            primHitsPrevFrame.GetDevicePtr<IntersectionBuffer>(),
            currentRays.GetDevicePtr<IntersectionRayBatch>(),
            currentHits.GetDevicePtr<IntersectionBuffer>(),
            secondaryRays.GetDevicePtr<IntersectionRayBatch>(),
            m_ShadowRayBatch->GetDevicePtr<ShadowRayBatch>(),
            m_LightBufferTemp->GetDevicePtr<LightDataBuffer>(),
            nullptr, //TODO: REPLACE THIS WITH LIGHT BUFFER FROM SCENE.
            m_ResultBuffer->GetDevicePtr<ResultBuffer>()); 

        Shade(shadingLaunchParams);

    }

    

    /*ResolveShadowRaysLaunchParameters optixShadowRaysLaunchParams{};

    optixShadowRaysLaunchParams.m_Common.m_ResolutionAndDepth = resolutionAndDepth;
    optixShadowRaysLaunchParams.m_Common.m_Traversable = optixRaysLaunchParams.m_Common.m_Traversable;
    optixShadowRaysLaunchParams.m_ShadowRays = m_ShadowRayBatch->GetDevicePtr<ShadowRayBatch>();
    optixShadowRaysLaunchParams.m_Results = m_ResultBuffer->GetDevicePtr<ResultBuffer>();

    m_PipelineShadowRaysLaunchParams->Write(optixShadowRaysLaunchParams);*/

   /* OptixShaderBindingTable SBT = m_ShadowRaysSBTGenerator->GetTableDesc();*/

    /*void* pixelBuffMultiChannelCuPtr = reinterpret_cast<void*>(*(*m_PixelBufferMultiChannel));
    SavePixelBufferToBMP(
        pixelBuffMultiChannelCuPtr,
        m_RenderResolution.x,
        m_RenderResolution.y,
        m_MaxDepth,
        frameSaveFilePath + "MultiPixelBuffer/",
        "MultiChannelPixelBuffer-BeforeLaunch");*/

    //Trace buffer of shadow rays using Optix ResolveShadowRays.
   /* CHECKOPTIXRESULT(optixLaunch(
        m_PipelineShadowRays,
        0,
        *(*m_PipelineShadowRaysLaunchParams),
        m_PipelineShadowRaysLaunchParams->GetSize(),
        &SBT,
        m_RenderResolution.x,
        m_RenderResolution.y,
        m_MaxDepth));
     
    cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;*/

    /*SavePixelBufferToBMP(
        pixelBuffMultiChannelCuPtr,
        m_RenderResolution.x,
        m_RenderResolution.y,
        m_MaxDepth,
        frameSaveFilePath + "MultiPixelBuffer/",
        "MultiChannelPixelBuffer-AfterLaunch");*/
    
    PostProcessLaunchParameters postProcessLaunchParams(
        m_RenderResolution,
        m_OutputResolution,
        m_ResultBuffer->GetDevicePtr<ResultBuffer>(),
        m_PixelBufferSingleChannel->GetDevicePtr<PixelBuffer>(),
        m_OutputBuffer->GetDevicePointer());

    /*SavePixelBufferToBMP(
        pixelBuffMultiChannelCuPtr,
        m_RenderResolution.x,
        m_RenderResolution.y,
        m_MaxDepth,
        frameSaveFilePath + "MultiPixelBuffer/",
        "MultiChannelPixelBuffer-AfterPostProcess");*/

    //Post processing using CUDA kernel.
    PostProcess(postProcessLaunchParams);

    ++m_FrameCount;
	
    //Return output image.
    return m_OutputBuffer->GetTexture();

}



void WaveFrontRenderer::GetRayBatchIndices(
    unsigned a_WaveIndex, 
    const std::array<unsigned, s_NumRayBatchTypes>& a_CurrentIndices, 
    std::array<unsigned, s_NumRayBatchTypes>& a_Indices)
{

    ////Create a copy of the current indices so that even if the references point to the same data it works.
    //const std::array<unsigned, s_NumRayBatchTypes> tempCurrentIndices = a_CurrentIndices;

    //GetPrimRayBatchIndex(a_WaveIndex, tempCurrentIndices, a_Indices);
    //GetCurrentRayBatchIndex(a_WaveIndex, tempCurrentIndices, a_Indices);
    //GetSecondaryRayBatchIndex(a_WaveIndex, tempCurrentIndices, a_Indices);

    const unsigned lastIndex = s_NumRayBatchTypes - 1;
    const unsigned primRayBatchIndex = a_CurrentIndices[static_cast<unsigned>(RayBatchTypeIndex::PRIM_RAYS_PREV_FRAME)];
    const unsigned currentRayBatchIndex = a_CurrentIndices[static_cast<unsigned>(RayBatchTypeIndex::CURRENT_RAYS)];
    const unsigned secondaryRayBatchIndex = a_CurrentIndices[static_cast<unsigned>(RayBatchTypeIndex::SECONDARY_RAYS)];
    const bool primRayBatchUsesLastIndex = primRayBatchIndex == lastIndex;

    if(a_WaveIndex == 0)
    {
        //During wave 0, use the opposite buffer of the Prim Ray Batch in order to generate camera rays.
        if(primRayBatchUsesLastIndex)
        {
            a_Indices[static_cast<unsigned>(RayBatchTypeIndex::CURRENT_RAYS)] = 0;
            a_Indices[static_cast<unsigned>(RayBatchTypeIndex::SECONDARY_RAYS)] = 1;
        }
        else
        {
            a_Indices[static_cast<unsigned>(RayBatchTypeIndex::CURRENT_RAYS)] = lastIndex;
            a_Indices[static_cast<unsigned>(RayBatchTypeIndex::SECONDARY_RAYS)] = lastIndex - 1;
        }

        //Prim Ray Batch stays the same during wave 0.
        a_Indices[static_cast<unsigned>(RayBatchTypeIndex::PRIM_RAYS_PREV_FRAME)] = primRayBatchIndex;

    }

    else if(a_WaveIndex == 1)
    {

        //Prim Ray Batch changes to the Current Ray Batch from wave 0 at wave 1.
        a_Indices[static_cast<unsigned>(RayBatchTypeIndex::PRIM_RAYS_PREV_FRAME)] = currentRayBatchIndex;

        //During wave 1 the Secondary Ray Batch from wave 0 becomes the Current Ray Batch.
        a_Indices[static_cast<unsigned>(RayBatchTypeIndex::CURRENT_RAYS)] = secondaryRayBatchIndex;

        //During wave 1 the Primary Ray Batch changes and the original batch becomes the Secondary Ray Batch.
        a_Indices[static_cast<unsigned>(RayBatchTypeIndex::SECONDARY_RAYS)] = primRayBatchIndex;

    }

    else
    {

        //Prim Ray Batch stays the same after wave 1.
        a_Indices[static_cast<unsigned>(RayBatchTypeIndex::PRIM_RAYS_PREV_FRAME)] = primRayBatchIndex;

        //During wave 1+ the Secondary Ray Batch from the previous wave becomes the Current Ray Batch and vice versa to swap the batches around.
        a_Indices[static_cast<unsigned>(RayBatchTypeIndex::CURRENT_RAYS)] = secondaryRayBatchIndex;
        a_Indices[static_cast<unsigned>(RayBatchTypeIndex::SECONDARY_RAYS)] = currentRayBatchIndex;

    }



}

void WaveFrontRenderer::GetHitBufferIndices(
    unsigned a_WaveIndex, 
    const std::array<unsigned, s_NumHitBufferTypes>& a_CurrentIndices, 
    std::array<unsigned, s_NumHitBufferTypes>& a_Indices)
{

    //At second wave, swap around the buffer indices. The CurrentHits buffer from wave 0 becomes the new PrimHitsPrevFrame buffer.
    if(a_WaveIndex == 1)
    {

        const unsigned primHitBufferIndex = a_CurrentIndices[static_cast<unsigned>(HitBufferTypeIndex::PRIM_HITS_PREV_FRAME)];
        const unsigned currentHitBufferIndex = a_CurrentIndices[static_cast<unsigned>(HitBufferTypeIndex::CURRENT_HITS)];

        a_Indices[static_cast<unsigned>(HitBufferTypeIndex::PRIM_HITS_PREV_FRAME)] = currentHitBufferIndex;
        a_Indices[static_cast<unsigned>(HitBufferTypeIndex::CURRENT_HITS)] = primHitBufferIndex;

    }
    else
    {

        a_Indices = a_CurrentIndices;

    }

}









std::unique_ptr<MemoryBuffer> WaveFrontRenderer::InterleaveVertexData(const PrimitiveData& a_MeshData)
{
    std::vector<Vertex> vertices;

    for (size_t i = 0; i < a_MeshData.m_Positions.Size(); i++)
    {
        auto& v = vertices.emplace_back();
        v.m_Position = make_float3(a_MeshData.m_Positions[i].x, a_MeshData.m_Positions[i].y, a_MeshData.m_Positions[i].z);
        if (!a_MeshData.m_TexCoords.Empty())
            v.m_UVCoord = make_float2(a_MeshData.m_TexCoords[i].x, a_MeshData.m_TexCoords[i].y);
        if (!a_MeshData.m_Normals.Empty())
            v.m_Normal = make_float3(a_MeshData.m_Normals[i].x, a_MeshData.m_Normals[i].y, a_MeshData.m_Normals[i].z);
    }
    return std::make_unique<MemoryBuffer>(vertices);
}

std::shared_ptr<Lumen::ILumenTexture> WaveFrontRenderer::CreateTexture(void* a_PixelData, uint32_t a_Width, uint32_t a_Height)
{

    static cudaChannelFormatDesc formatDesc = cudaCreateChannelDesc<uchar4>();
    return std::make_shared<Texture>(a_PixelData, formatDesc, a_Width, a_Height);

}

std::unique_ptr<Lumen::ILumenPrimitive> WaveFrontRenderer::CreatePrimitive(PrimitiveData& a_PrimitiveData)
{
    auto vertexBuffer = InterleaveVertexData(a_PrimitiveData);
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();

    std::vector<uint32_t> correctedIndices;

    if (a_PrimitiveData.m_IndexSize != 4)
    {
        VectorView<uint16_t, uint8_t> indexView(a_PrimitiveData.m_IndexBinary);

        for (size_t i = 0; i < indexView.Size(); i++)
        {
            correctedIndices.push_back(indexView[i]);
        }

    }

	//printf("Index buffer Size %i \n", static_cast<int>(correctedIndices.size()));
    std::unique_ptr<MemoryBuffer> indexBuffer = std::make_unique<MemoryBuffer>(correctedIndices);

    unsigned int geomFlags = OPTIX_GEOMETRY_FLAG_NONE;

    OptixAccelBuildOptions buildOptions = {};
    buildOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
    buildOptions.motionOptions = {};

    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    buildInput.triangleArray.indexBuffer = **indexBuffer;
    buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    buildInput.triangleArray.indexStrideInBytes = 0;
    buildInput.triangleArray.numIndexTriplets = correctedIndices.size() / 3;
    buildInput.triangleArray.vertexBuffers = &**vertexBuffer;
    buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexStrideInBytes = sizeof(Vertex);
    buildInput.triangleArray.numVertices = a_PrimitiveData.m_Positions.Size();
    buildInput.triangleArray.numSbtRecords = 1;
    buildInput.triangleArray.flags = &geomFlags;

    auto gAccel = BuildGeometryAccelerationStructure(buildOptions, buildInput);

    auto prim = std::make_unique<PTPrimitive>(std::move(vertexBuffer), std::move(indexBuffer), std::move(gAccel));

    prim->m_Material = a_PrimitiveData.m_Material;

    prim->m_RecordHandle = m_RaysSBTGenerator->AddHitGroup<DevicePrimitive>();
    auto& rec = prim->m_RecordHandle.GetRecord();
    rec.m_Header = GetProgramGroupHeader(s_RaysHitPGName);
    rec.m_Data.m_VertexBuffer = prim->m_VertBuffer->GetDevicePtr<Vertex>();
    rec.m_Data.m_IndexBuffer = prim->m_IndexBuffer->GetDevicePtr<unsigned int>();
    rec.m_Data.m_Material = reinterpret_cast<Material*>(prim->m_Material.get())->GetDeviceMaterial();

    /*printf("Primitive: Material: %p, VertexBuffer: %p, IndexBufferPtr: %p \n",
        rec.m_Data.m_Material, 
        rec.m_Data.m_VertexBuffer, 
        rec.m_Data.m_IndexBuffer);*/

    return prim;
}

std::shared_ptr<Lumen::ILumenMesh> WaveFrontRenderer::CreateMesh(
    std::vector<std::unique_ptr<Lumen::ILumenPrimitive>>& a_Primitives)
{
    auto mesh = std::make_shared<PTMesh>(a_Primitives, m_ServiceLocator);
    return mesh;
}

std::shared_ptr<Lumen::ILumenMaterial> WaveFrontRenderer::CreateMaterial(const MaterialData& a_MaterialData)
{

    auto mat = std::make_shared<Material>();
    mat->SetDiffuseColor(a_MaterialData.m_DiffuseColor);
    mat->SetDiffuseTexture(a_MaterialData.m_DiffuseTexture);
    mat->SetEmission(a_MaterialData.m_EmssivionVal);
	
    return mat;

}

std::shared_ptr<Lumen::ILumenScene> WaveFrontRenderer::CreateScene(SceneData a_SceneData)
{
    return std::make_shared<PTScene>(a_SceneData, m_ServiceLocator);
}

std::shared_ptr<Lumen::ILumenVolume> WaveFrontRenderer::CreateVolume(const std::string& a_FilePath)
{
    std::shared_ptr<Lumen::ILumenVolume> volume = std::make_shared<PTVolume>(a_FilePath, m_ServiceLocator);

    return volume;
}

#endif