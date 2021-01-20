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

#include <cstdio>
#include <fstream>
#include <sstream>
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
m_TempBuffers(),
m_RayBatchIndices({0}),
m_ResultBuffer(nullptr),
m_PixelBuffer3Channels(nullptr),
m_PixelBuffer1Channel(nullptr),
m_RayBatches(),
m_IntersectionBuffers(),
m_ShadowRayBatch(nullptr),
m_LightBufferTemp(nullptr),
m_Texture(nullptr),
m_Resolution({0u, 0u}),
m_MaxDepth(0u),
m_Initialized(false)
{

    m_Initialized = Initialize(a_InitializationData);
    if(!m_Initialized)
    {
        std::fprintf(stderr, "Initialization of wavefront renderer unsuccessful");
        return;
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

    m_Resolution = a_InitializationData.m_Resolution;
    //m_MaxDepth = a_InitializationData.m_MaxDepth;
    m_MaxDepth = 5;

    InitializeContext();
    success &= CreatePipelines(shaderPath);
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

    const std::string resolveRaysParams = "resolveRaysParams";
    const std::string resolveRaysRayGenFuncName = "__raygen__ResolveRaysRayGen";
    const std::string resolveRaysHitFuncName = "__closesthit__ResolveRaysClosestHit";
    const std::string resolveShadowRaysParams = "resolveShadowRaysParams";
    const std::string resolveShadowRaysRayGenFuncName = "__raygen__ResolveShadowRaysRayGen";
    const std::string resolveShadowRaysHitFuncName = "__anyhit__ResolveShadowRaysAnyHit";

    OptixPipelineCompileOptions compileOptions = CreatePipelineOptions(resolveRaysParams, 2, 2);

    OptixModule shaderModule = CreateModule(a_ShaderPath, compileOptions);
    if (shaderModule == nullptr) { return false; }

    success &= CreatePipeline(
        shaderModule,
        compileOptions,
        PipelineType::RESOLVE_RAYS, 
        resolveRaysRayGenFuncName, 
        resolveRaysHitFuncName, 
        m_PipelineRays);

    compileOptions = CreatePipelineOptions(resolveShadowRaysParams, 1, 2);
    shaderModule = CreateModule(a_ShaderPath, compileOptions);
    if (shaderModule == nullptr) { return false; }

    success &= CreatePipeline(
        shaderModule,
        compileOptions,
        PipelineType::RESOLVE_SHADOW_RAYS, 
        resolveShadowRaysRayGenFuncName, 
        resolveShadowRaysHitFuncName, 
        m_PipelineShadowRays);

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
    OptixPipeline& a_Pipeline)
{

    OptixProgramGroup rayGenProgram = nullptr;
    OptixProgramGroup hitProgram = nullptr;

    OptixProgramGroupDesc rgGroupDesc = {};
    rgGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rgGroupDesc.raygen.entryFunctionName = a_RayGenFuncName.c_str();
    rgGroupDesc.raygen.module = a_Module;

    OptixProgramGroupDesc htGroupDesc = {};
    htGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

    switch (a_Type)
    {

        case PipelineType::RESOLVE_RAYS:
            {
                htGroupDesc.hitgroup.entryFunctionNameCH = a_HitFuncName.c_str();
                htGroupDesc.hitgroup.moduleCH = a_Module;

                rayGenProgram = CreateProgramGroup(rgGroupDesc, s_RaysRayGenPGName);
                hitProgram = CreateProgramGroup(htGroupDesc, s_RaysHitPGName);

                break;
            }

        case PipelineType::RESOLVE_SHADOW_RAYS:
            {
                htGroupDesc.hitgroup.entryFunctionNameAH = a_HitFuncName.c_str();
                htGroupDesc.hitgroup.moduleAH = a_Module;

                rayGenProgram = CreateProgramGroup(rgGroupDesc, s_ShadowRaysRayGenPGName);
                hitProgram = CreateProgramGroup(htGroupDesc, s_ShadowRaysHitPGName);

                break;
            }
        default:
            {
                rayGenProgram = nullptr;
            }
            break;
    }

    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    pipelineLinkOptions.maxTraceDepth = 1;

    OptixProgramGroup programGroups[] = { rayGenProgram, hitProgram };

    char log[2048];
    auto logSize = sizeof(log);

    OptixResult error{};

    CHECKOPTIXRESULT(error = optixPipelineCreate(
        m_DeviceContext,
        &a_PipelineOptions,
        &pipelineLinkOptions,
        programGroups,
        sizeof(programGroups) / sizeof(OptixProgramGroup),
        log,
        &logSize,
        &a_Pipeline));

    if (error) { return false; }

    OptixStackSizes stackSizes = {};

    AccumulateStackSizes(rayGenProgram, stackSizes);
    AccumulateStackSizes(hitProgram, stackSizes);

    auto finalSizes = ComputeStackSizes(stackSizes, 1, 0, 0);

    CHECKOPTIXRESULT(error = optixPipelineSetStackSize(
        a_Pipeline,
        finalSizes.DirectSizeTrace,
        finalSizes.DirectSizeState,
        finalSizes.ContinuationSize,
        3));

    if (error) { return false; }

    return true;

}

void WaveFrontRenderer::CreatePipelineBuffers()
{

    m_PipelineRaysLaunchParams = std::make_unique<MemoryBuffer>(sizeof(WaveFront::ResolveRaysLaunchParameters));
    m_PipelineShadowRaysLaunchParams = std::make_unique<MemoryBuffer>(sizeof(WaveFront::ResolveShadowRaysLaunchParameters));

}

OptixModule WaveFrontRenderer::CreateModule(const std::string& a_PtxPath, const OptixPipelineCompileOptions& a_PipelineOptions) const
{

    OptixModuleCompileOptions moduleOptions = {};
    moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
    moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;

    std::ifstream stream;
    stream.open(a_PtxPath);

    if(!stream.is_open())
    {
        return nullptr;
    }

    std::stringstream stringStream;
    stringStream << stream.rdbuf();
    std::string source = stringStream.str();

    char log[2048];
    auto logSize = sizeof(log);
    OptixModule module;

    CHECKOPTIXRESULT(optixModuleCreateFromPTX(
        m_DeviceContext, 
        &moduleOptions, 
        &a_PipelineOptions, 
        source.c_str(), 
        source.size(), 
        log, 
        &logSize, 
        &module));

    return module;

}

OptixProgramGroup WaveFrontRenderer::CreateProgramGroup(OptixProgramGroupDesc a_ProgramGroupDesc, const std::string& a_Name)
{

    OptixProgramGroupOptions programGroupOptions = {};

    char log[2048];
    auto logSize = sizeof(log);

    OptixProgramGroup programGroup;

    CHECKOPTIXRESULT(optixProgramGroupCreate(
        m_DeviceContext, 
        &a_ProgramGroupDesc, 
        1, 
        &programGroupOptions, 
        log, 
        &logSize, 
        &programGroup));

    m_ProgramGroups.emplace(a_Name, programGroup);

    return programGroup;

}

std::unique_ptr<AccelerationStructure> WaveFrontRenderer::BuildGeometryAccelerationStructure(
    const OptixAccelBuildOptions& a_BuildOptions, 
    const OptixBuildInput& a_BuildInput)
{

    // Let Optix compute how much memory the output buffer and the temporary buffer need to have, then create these buffers
    OptixAccelBufferSizes sizes;
    CHECKOPTIXRESULT(optixAccelComputeMemoryUsage(m_DeviceContext, &a_BuildOptions, &a_BuildInput, 1, &sizes));
    MemoryBuffer tempBuffer(sizes.tempSizeInBytes);
    std::unique_ptr<MemoryBuffer> resultBuffer = std::make_unique<MemoryBuffer>(sizes.outputSizeInBytes);

    OptixTraversableHandle handle;
    // It is possible to have functionality similar to DX12 Command lists with cuda streams, though it is not required.
    // Just worth mentioning if we want that functionality.
    CHECKOPTIXRESULT(optixAccelBuild(m_DeviceContext, 0, &a_BuildOptions, &a_BuildInput, 1, *tempBuffer, sizes.tempSizeInBytes,
        **resultBuffer, sizes.outputSizeInBytes, &handle, nullptr, 0));

    // Needs to return the resulting memory buffer and the traversable handle
    return std::make_unique<AccelerationStructure>(handle, std::move(resultBuffer));

}

std::unique_ptr<AccelerationStructure> WaveFrontRenderer::BuildInstanceAccelerationStructure(std::vector<OptixInstance> a_Instances)
{


    auto instanceBuffer = MemoryBuffer(a_Instances.size() * sizeof(OptixInstance));
    instanceBuffer.Write(a_Instances.data(), a_Instances.size() * sizeof(OptixInstance), 0);
    OptixBuildInput buildInput = {};

    assert(*instanceBuffer % OPTIX_INSTANCE_BYTE_ALIGNMENT == 0);

    buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    buildInput.instanceArray.instances = *instanceBuffer;
    buildInput.instanceArray.numInstances = static_cast<uint32_t>(a_Instances.size());
    buildInput.instanceArray.aabbs = 0;
    buildInput.instanceArray.numAabbs = 0;

    OptixAccelBuildOptions buildOptions = {};
    // Based on research, it is more efficient to continuously be rebuilding most instance acceleration structures rather than to update them
    // This is because updating an acceleration structure reduces its quality, thus lowering the ray traversal speed through it
    // while rebuilding will preserve this quality. Since instance acceleration structures are cheap to build, it is faster to rebuild them than deal
    // with the lower traversal speed
    buildOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
    buildOptions.motionOptions.numKeys = 0; // No motion

    OptixAccelBufferSizes sizes;
    CHECKOPTIXRESULT(optixAccelComputeMemoryUsage(
        m_DeviceContext,
        &buildOptions,
        &buildInput,
        1,
        &sizes));

    auto tempBuffer = MemoryBuffer(sizes.tempSizeInBytes);
    auto resultBuffer = std::make_unique<MemoryBuffer>(sizes.outputSizeInBytes);

    OptixTraversableHandle handle;
    CHECKOPTIXRESULT(optixAccelBuild(
        m_DeviceContext,
        0,
        &buildOptions,
        &buildInput,
        1,
        *tempBuffer,
        sizes.tempSizeInBytes,
        **resultBuffer,
        sizes.outputSizeInBytes,
        &handle,
        nullptr,
        0));

    return std::make_unique<AccelerationStructure>(handle, std::move(resultBuffer));;
    
}

//TODO add necessary data to shaderBindingTable.
void WaveFrontRenderer::CreateShaderBindingTables()
{

    //Do these need a data struct if there is no data needed per "shader"??
    
    m_RaysRayGenRecord = m_RaysSBTGenerator->SetRayGen<void>();
    m_RaysHitRecord = m_RaysSBTGenerator->AddHitGroup<void>();
    m_RaysMissRecord = m_RaysSBTGenerator->AddMiss<void>();

    auto& raysRayGenRecord = m_RaysRayGenRecord.GetRecord();
    raysRayGenRecord.m_Header = GetProgramGroupHeader(s_RaysRayGenPGName);

    auto& raysHitRecord = m_RaysHitRecord.GetRecord();
    raysHitRecord.m_Header = GetProgramGroupHeader(s_RaysHitPGName);

    m_ShadowRaysRayGenRecord = m_ShadowRaysSBTGenerator->SetRayGen<void>();
    m_ShadowRaysHitRecord = m_ShadowRaysSBTGenerator->AddHitGroup<void>();
    m_ShadowRaysMissRecord = m_ShadowRaysSBTGenerator->AddMiss<void>();

    auto& shadowRaysRayGenRecord = m_ShadowRaysRayGenRecord.GetRecord();
    shadowRaysRayGenRecord.m_Header = GetProgramGroupHeader(s_ShadowRaysRayGenPGName);

    auto& shadowRaysHitRecord = m_ShadowRaysHitRecord.GetRecord();
    shadowRaysHitRecord.m_Header = GetProgramGroupHeader(s_ShadowRaysHitPGName);
   
}

void WaveFrontRenderer::CreateOutputBuffer()
{

    m_OutputBuffer = std::make_unique<::OutputBuffer>(m_Resolution.x, m_Resolution.y);

}

void WaveFrontRenderer::CreateDataBuffers()
{

    const unsigned numPixels = static_cast<unsigned>(m_Resolution.x) * static_cast<unsigned>(m_Resolution.y);

    const unsigned raysPerPixel = 1;
    const unsigned shadowRaysPerPixel = 1;
    const unsigned maxDepth = 3;
    const unsigned numOutputChannels = ResultBuffer::s_NumOutputChannels;

    //Allocate pixel buffer.
    m_PixelBuffer3Channels = std::make_unique<MemoryBuffer>(sizeof(PixelBuffer) + numPixels * numOutputChannels * sizeof(float3));
    m_PixelBuffer3Channels->Write(numPixels, 0);
    m_PixelBuffer3Channels->Write(numOutputChannels, sizeof(PixelBuffer::m_NumPixels));

    m_PixelBuffer1Channel = std::make_unique<MemoryBuffer>(sizeof(PixelBuffer) + numPixels * 1 * sizeof(float3));
    m_PixelBuffer1Channel->Write(numPixels, 0);
    m_PixelBuffer1Channel->Write(1, sizeof(PixelBuffer::m_NumPixels));

    const PixelBuffer* pixelBufferPtr = m_PixelBuffer3Channels->GetDevicePtr<PixelBuffer>();

    //Allocate result buffer.
    m_ResultBuffer = std::make_unique<MemoryBuffer>(sizeof(ResultBuffer));
    m_ResultBuffer->Write(pixelBufferPtr, 0);

    const unsigned rayBatchEmptySize = sizeof(RayBatch);
    const unsigned rayDataStructSize = sizeof(RayData);

    //Allocate and initialize ray batches.
    for(auto& rayBatch : m_RayBatches)
    {
        rayBatch = std::make_unique<MemoryBuffer>(
            static_cast<size_t>(rayBatchEmptySize) + 
            static_cast<size_t>(numPixels) * 
            static_cast<size_t>(raysPerPixel) * 
            static_cast<size_t>(rayDataStructSize));
        rayBatch->Write(numPixels, 0);
        rayBatch->Write(raysPerPixel, sizeof(RayBatch::m_NumPixels));
    }

    const unsigned intersectionBufferEmptySize = sizeof(IntersectionBuffer);
    const unsigned intersectionDataStructSize = sizeof(IntersectionData);

    for(auto& intersectionBuffer : m_IntersectionBuffers)
    {
        intersectionBuffer = std::make_unique<MemoryBuffer>(
           static_cast<size_t>(intersectionBufferEmptySize) +
           static_cast<size_t>(numPixels) * 
           static_cast<size_t>(raysPerPixel) * 
           static_cast<size_t>(intersectionDataStructSize));
        intersectionBuffer->Write(numPixels, 0);
        intersectionBuffer->Write(raysPerPixel, sizeof(IntersectionBuffer::m_NumPixels));
    }

    const unsigned ShadowRayBatchEmptySize = sizeof(ShadowRayBatch);
    const unsigned ShadowRayDataStructSize = sizeof(ShadowRayData);

    m_ShadowRayBatch = std::make_unique<MemoryBuffer>(
        static_cast<size_t>(ShadowRayBatchEmptySize) + 
        static_cast<size_t>(maxDepth) * 
        static_cast<size_t>(numPixels) * 
        static_cast<size_t>(shadowRaysPerPixel) * 
        static_cast<size_t>(ShadowRayDataStructSize));
    m_ShadowRayBatch->Write(maxDepth, 0);
    m_ShadowRayBatch->Write(numPixels, sizeof(ShadowRayBatch::m_MaxDepth));
    m_ShadowRayBatch->Write(shadowRaysPerPixel, sizeof(ShadowRayBatch::m_MaxDepth) + sizeof(ShadowRayBatch::m_NumPixels));



    const unsigned LightBufferEmptySize = sizeof(LightBuffer);

    m_LightBufferTemp = std::make_unique<MemoryBuffer>(0);

}

void WaveFrontRenderer::SetupInitialBufferIndices()
{

    m_RayBatchIndices[static_cast<unsigned>(RayBatchTypeIndex::PRIM_RAYS_PREV_FRAME)] = 0;
    m_RayBatchIndices[static_cast<unsigned>(RayBatchTypeIndex::CURRENT_RAYS)] = s_NumRayBatchTypes - 1;
    m_RayBatchIndices[static_cast<unsigned>(RayBatchTypeIndex::SECONDARY_RAYS)] = 1;

    m_HitBufferIndices[static_cast<unsigned>(HitBufferTypeIndex::PRIM_HITS_PREV_FRAME)] = 0;
    m_HitBufferIndices[static_cast<unsigned>(HitBufferTypeIndex::CURRENT_HITS)] = s_NumHitBufferTypes - 1;
}



void WaveFrontRenderer::DebugCallback(unsigned a_Level, const char* a_Tag, const char* a_Message, void*)
{

    std::printf("%u::%s:: %s\n\n", a_Level, a_Tag, a_Message);

}

void WaveFrontRenderer::AccumulateStackSizes(OptixProgramGroup a_ProgramGroup, OptixStackSizes& a_StackSizes)
{

    OptixStackSizes localSizes;
    CHECKOPTIXRESULT(optixProgramGroupGetStackSize(a_ProgramGroup, &localSizes));
    a_StackSizes.cssRG = std::max(a_StackSizes.cssRG, localSizes.cssRG);
    a_StackSizes.cssMS = std::max(a_StackSizes.cssMS, localSizes.cssMS);
    a_StackSizes.cssIS = std::max(a_StackSizes.cssIS, localSizes.cssIS);
    a_StackSizes.cssAH = std::max(a_StackSizes.cssAH, localSizes.cssAH);
    a_StackSizes.cssCH = std::max(a_StackSizes.cssCH, localSizes.cssCH);
    a_StackSizes.cssCC = std::max(a_StackSizes.cssCC, localSizes.cssCC);
    a_StackSizes.dssDC = std::max(a_StackSizes.dssDC, localSizes.dssDC);

}

ProgramGroupHeader WaveFrontRenderer::GetProgramGroupHeader(const std::string& a_GroupName) const
{

    assert(m_ProgramGroups.count(a_GroupName) != 0);

    ProgramGroupHeader header{};

    CHECKOPTIXRESULT(optixSbtRecordPackHeader(m_ProgramGroups.at(a_GroupName), &header));

    return header;

}

GLuint WaveFrontRenderer::TraceFrame()
{

    CHECKLASTCUDAERROR;

    //Generate Camera rays using CUDA kernel.
    float3 eye, u, v, w;
    m_Camera.GetVectorData(eye, u, v, w);
    const WaveFront::DeviceCameraData cameraData(eye, u, v, w);

    //Get new Ray Batch to fill with Primary Rays (Either first or last ray batch, opposite of current PrimRaysPrevFrame batch)
    std::array<unsigned, s_NumRayBatchTypes> batchIndices{};
    GetRayBatchIndices(0, m_RayBatchIndices, batchIndices);
    MemoryBuffer& currentRaysBatch = *m_RayBatches[batchIndices[static_cast<unsigned>(RayBatchTypeIndex::CURRENT_RAYS)]];

    //Generate primary rays using the setup parameters
    const WaveFront::SetupLaunchParameters setupParams(m_Resolution, cameraData, currentRaysBatch.GetDevicePtr<RayBatch>());
    GenerateRays(setupParams);

    //Initialize resolveRaysLaunchParameters with common variables between different waves.
    WaveFront::ResolveRaysLaunchParameters optixRaysLaunchParams{};
    optixRaysLaunchParams.m_Common.m_Traversable = dynamic_cast<PTScene&>(*m_Scene).GetSceneAccelerationStructure(); //TODO: make sure if scene is empty it doesn't result in errors.

    uint3 resolutionAndDepth = make_uint3(m_Resolution.x, m_Resolution.y, 0);

    //Loop
    //Trace buffer of rays using Optix ResolveRays pipeline
    //Calculate shading for intersections in buffer using CUDA kernel.
    for(unsigned waveIndex = 0; waveIndex < m_MaxDepth; ++waveIndex)
    {

        GetRayBatchIndices(waveIndex, m_RayBatchIndices, m_RayBatchIndices);
        GetHitBufferIndices(waveIndex, m_HitBufferIndices, m_HitBufferIndices);

        MemoryBuffer& primRaysPrevFrame =   *m_RayBatches[m_RayBatchIndices[static_cast<unsigned>(RayBatchTypeIndex::PRIM_RAYS_PREV_FRAME)]];
        MemoryBuffer& currentRays =         *m_RayBatches[m_RayBatchIndices[static_cast<unsigned>(RayBatchTypeIndex::CURRENT_RAYS)]];
        MemoryBuffer& secondaryRays =       *m_RayBatches[m_RayBatchIndices[static_cast<unsigned>(RayBatchTypeIndex::SECONDARY_RAYS)]];

        MemoryBuffer& primHitsPrevFrame =   *m_IntersectionBuffers[m_HitBufferIndices[static_cast<unsigned>(HitBufferTypeIndex::PRIM_HITS_PREV_FRAME)]];
        MemoryBuffer& currentHits =         *m_IntersectionBuffers[m_HitBufferIndices[static_cast<unsigned>(HitBufferTypeIndex::CURRENT_HITS)]];


        /*printf(
            "Wave %i: Indices: %i %i %i \n",
            WaveIndex,
            m_RayBatchIndices[static_cast<unsigned>(RayBatchTypeIndex::PRIM_RAYS_PREV_FRAME)],
            m_RayBatchIndices[static_cast<unsigned>(RayBatchTypeIndex::CURRENT_RAYS)],
            m_RayBatchIndices[static_cast<unsigned>(RayBatchTypeIndex::SECONDARY_RAYS)]);*/

        /*printf(
            "Wave %i: Hit Buffer Indices: %i %i \n",
            waveIndex,
            m_HitBufferIndices[static_cast<unsigned>(HitBufferTypeIndex::PRIM_HITS_PREV_FRAME)],
            m_HitBufferIndices[static_cast<unsigned>(HitBufferTypeIndex::CURRENT_HITS)]);*/

        //Resolution and current depth(, current depth = current wave index)
        resolutionAndDepth.z = waveIndex;

        optixRaysLaunchParams.m_Common.m_ResolutionAndDepth = resolutionAndDepth;
        optixRaysLaunchParams.m_Rays = currentRays.GetDevicePtr<RayBatch>();
        optixRaysLaunchParams.m_Intersections = currentHits.GetDevicePtr<IntersectionBuffer>();

        m_PipelineRaysLaunchParams->Write(optixRaysLaunchParams);

        OptixShaderBindingTable SBT = m_RaysSBTGenerator->GetTableDesc();

        //Launch OptiX ResolveRays pipeline to resolve all of the rays in Current Rays Batch (Secondary ray batch from previous wave).
        optixLaunch(
            m_PipelineRays,
            0,
            *(*m_PipelineRaysLaunchParams),
            m_PipelineRaysLaunchParams->GetSize(),
            &SBT,
            m_Resolution.x,
            m_Resolution.y,
            1); //Depth = 1 as the waves only trace one bounce per wave.

        WaveFront::ShadingLaunchParameters shadingLaunchParams(
            resolutionAndDepth, 
            primRaysPrevFrame.GetDevicePtr<RayBatch>(),
            primHitsPrevFrame.GetDevicePtr<IntersectionBuffer>(),
            currentRays.GetDevicePtr<RayBatch>(),
            currentHits.GetDevicePtr<IntersectionBuffer>(),
            secondaryRays.GetDevicePtr<RayBatch>(),
            m_ShadowRayBatch->GetDevicePtr<ShadowRayBatch>(),
            m_LightBufferTemp->GetDevicePtr<LightBuffer>()); //TODO: REPLACE THIS WITH LIGHT BUFFER FROM SCENE.

        Shade(shadingLaunchParams);

    }

    

    ResolveShadowRaysLaunchParameters optixShadowRaysLaunchParams{};

    optixShadowRaysLaunchParams.m_Common.m_ResolutionAndDepth = resolutionAndDepth;
    optixShadowRaysLaunchParams.m_Common.m_Traversable = dynamic_cast<PTScene&>(*m_Scene).GetSceneAccelerationStructure();
    optixShadowRaysLaunchParams.m_ShadowRays = m_ShadowRayBatch->GetDevicePtr<ShadowRayBatch>();
    optixShadowRaysLaunchParams.m_Results = m_ResultBuffer->GetDevicePtr<ResultBuffer>();

    m_PipelineShadowRaysLaunchParams->Write(optixShadowRaysLaunchParams);

    OptixShaderBindingTable SBT = m_ShadowRaysSBTGenerator->GetTableDesc();

    //Trace buffer of shadow rays using Optix ResolveShadowRays.
    optixLaunch(
        m_PipelineShadowRays,
        0,
        *(*m_PipelineShadowRaysLaunchParams),
        m_PipelineShadowRaysLaunchParams->GetSize(),
        &SBT,
        m_Resolution.x,
        m_Resolution.y,
        m_MaxDepth);


    
    PostProcessLaunchParameters postProcessLaunchParams(
        m_Resolution,
        m_Resolution,
        m_ResultBuffer->GetDevicePtr<ResultBuffer>(),
        m_PixelBuffer1Channel->GetDevicePtr<PixelBuffer>(),
        m_OutputBuffer->GetDevicePointer());

    //Post processing using CUDA kernel.
    PostProcess(postProcessLaunchParams);



    //Return output image.
    return m_OutputBuffer->GetTexture();

}

WaveFrontRenderer::ComputedStackSizes WaveFrontRenderer::ComputeStackSizes(
    OptixStackSizes a_StackSizes, 
    int a_TraceDepth, 
    int a_DirectDepth, 
    int a_ContinuationDepth)
{

    ComputedStackSizes sizes;
    sizes.DirectSizeState = a_StackSizes.dssDC * a_DirectDepth;
    sizes.DirectSizeTrace = a_StackSizes.dssDC * a_DirectDepth;

    unsigned int cssCCTree = a_ContinuationDepth * a_StackSizes.cssCC;

    unsigned int cssCHOrMSPlusCCTree = std::max(a_StackSizes.cssCH, a_StackSizes.cssMS) + cssCCTree;

    sizes.ContinuationSize =
        a_StackSizes.cssRG + cssCCTree +
        (std::max(a_TraceDepth, 1) - 1) * cssCHOrMSPlusCCTree +
        std::min(a_TraceDepth, 1) * std::max(cssCHOrMSPlusCCTree, a_StackSizes.cssIS + a_StackSizes.cssAH);

    return sizes;

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
    rec.m_Data.m_Material = static_cast<Material*>(prim->m_Material.get())->GetDeviceMaterial();

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