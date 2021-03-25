#ifdef WAVEFRONT
#include "OptixWrapper.h"
#include "CudaUtilities.h"
#include "MemoryBuffer.h"
#include "AccelerationStructure.h"
#include "ShaderBindingTableGen.h"

#include <Optix/optix_stubs.h>
#include <algorithm>
#include <fstream>
#include <sstream>

using namespace WaveFront;



OptixWrapper::OptixWrapper(const InitializationData& a_InitializationData)
{

    if (!Initialize(a_InitializationData))
    {
        std::fprintf(stderr, "Initialization of wavefront renderer unsuccessful");
        abort();
    }

}

OptixWrapper::~OptixWrapper()
{

    CHECKOPTIXRESULT(optixPipelineDestroy(m_Pipeline));
    CHECKOPTIXRESULT(optixModuleDestroy(m_Module));

    DestroyProgramGroups();

    CHECKOPTIXRESULT(optixDeviceContextDestroy(m_DeviceContext));

}



//Acceleration Structure building ---------

std::unique_ptr<AccelerationStructure> OptixWrapper::BuildGeometryAccelerationStructure(
    const OptixAccelBuildOptions& a_BuildOptions,
    const OptixBuildInput& a_BuildInput) const
{

    // Let Optix compute how much memory the output buffer and the temporary buffer need to have, then create these buffers
    OptixAccelBufferSizes sizes{};
    CHECKOPTIXRESULT(optixAccelComputeMemoryUsage(m_DeviceContext, &a_BuildOptions, &a_BuildInput, 1, &sizes));
    MemoryBuffer tempBuffer(sizes.tempSizeInBytes);
    std::unique_ptr<MemoryBuffer> resultBuffer = std::make_unique<MemoryBuffer>(sizes.outputSizeInBytes);

    OptixTraversableHandle handle;
    // It is possible to have functionality similar to DX12 Command lists with cuda streams, though it is not required.
    // Just worth mentioning if we want that functionality.
    CHECKOPTIXRESULT(
        optixAccelBuild(
            m_DeviceContext, 
            0, 
            &a_BuildOptions, 
            &a_BuildInput, 
            1, 
            *tempBuffer, 
            sizes.tempSizeInBytes,
            **resultBuffer, 
            sizes.outputSizeInBytes, 
            &handle, 
            nullptr, 
            0));

    // Needs to return the resulting memory buffer and the traversable handle
    return std::make_unique<AccelerationStructure>(handle, std::move(resultBuffer));

}

std::unique_ptr<AccelerationStructure> OptixWrapper::BuildInstanceAccelerationStructure(std::vector<OptixInstance> a_Instances) const
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



bool OptixWrapper::Initialize(const InitializationData& a_InitializationData)
{

    bool success = true;

    //Create SBT instance.
    m_SBTGenerator = std::make_unique<ShaderBindingTableGenerator>();

    success &= InitializeContext(a_InitializationData.m_CUDAContext);
    success &= CreatePipeline(a_InitializationData.m_ProgramData);

    SetupPipelineBuffer();
    SetupShaderBindingTable();

    return success;

}

bool OptixWrapper::InitializeContext(CUcontext a_CUDAContext)
{

    CHECKOPTIXRESULT(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &OptixDebugCallback;
    options.logCallbackLevel = 4;
    CHECKOPTIXRESULT(optixDeviceContextCreate(a_CUDAContext, &options, &m_DeviceContext));

    return true;

}

bool OptixWrapper::CreatePipeline(const InitializationData::ProgramData& a_ProgramData)
{

    const OptixPipelineCompileOptions compileOptions = CreatePipelineOptions(
        a_ProgramData.m_ProgramLaunchParamName,
        a_ProgramData.m_MaxNumPayloads,
        a_ProgramData.m_MaxNumHitResultAttributes);

    m_Module = CreateModule(a_ProgramData.m_ProgramPath, compileOptions);

    return CreatePipeline(
        m_Module,
        compileOptions,
        a_ProgramData.m_ProgramRayGenFuncName,
        a_ProgramData.m_ProgramMissFuncName,
        a_ProgramData.m_ProgramAnyHitFuncName,
        a_ProgramData.m_ProgramClosestHitFuncName,
        m_Pipeline);

}

OptixPipelineCompileOptions OptixWrapper::CreatePipelineOptions(
    const std::string& a_LaunchParamName,
    unsigned int a_NumPayloadValues,
    unsigned int a_NumAttributes) const
{

    OptixPipelineCompileOptions pipelineOptions = {};
    pipelineOptions.usesMotionBlur = false;
    pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipelineOptions.numPayloadValues = std::clamp(a_NumPayloadValues, 0u, 8u); //Move to initializationData.
    pipelineOptions.numAttributeValues = std::clamp(a_NumAttributes, 2u, 8u); //Move to initializationData.
    pipelineOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG;
    pipelineOptions.pipelineLaunchParamsVariableName = a_LaunchParamName.c_str(); //Move to initializationData.
    pipelineOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE & OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

    return pipelineOptions;

}

bool OptixWrapper::CreatePipeline(
    const OptixModule& a_Module,
    const OptixPipelineCompileOptions& a_PipelineOptions,
    const std::string& a_RayGenFuncName,
    const std::string& a_MissFuncName,
    const std::string& a_AnyHitFuncName,
    const std::string& a_ClosestHitFuncName,
    OptixPipeline& a_Pipeline)
{

    OptixProgramGroup rayGenProgram = nullptr;
    OptixProgramGroup hitProgram = nullptr;
    OptixProgramGroup missProgram = nullptr;

    OptixProgramGroupDesc rayGenGroupDesc = {};
    rayGenGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rayGenGroupDesc.raygen.entryFunctionName = a_RayGenFuncName.c_str();
    rayGenGroupDesc.raygen.module = a_Module;

    OptixProgramGroupDesc hitGroupDesc = {};
    hitGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitGroupDesc.hitgroup.entryFunctionNameAH = (a_AnyHitFuncName.length() > 0) ? a_AnyHitFuncName.c_str() : nullptr;
    hitGroupDesc.hitgroup.entryFunctionNameCH = (a_ClosestHitFuncName.length() > 0) ? a_ClosestHitFuncName.c_str() : nullptr;
    hitGroupDesc.hitgroup.moduleAH = a_Module;
    hitGroupDesc.hitgroup.moduleCH = a_Module;

    OptixProgramGroupDesc missGroupDesc = {};
    missGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missGroupDesc.miss.entryFunctionName = a_MissFuncName.c_str();
    missGroupDesc.miss.module = a_Module;

    rayGenProgram = CreateProgramGroup(rayGenGroupDesc, s_RayGenPGName);
    hitProgram    = CreateProgramGroup(hitGroupDesc,    s_HitPGName);
    missProgram   = CreateProgramGroup(missGroupDesc,   s_MissPGName);

    if (rayGenProgram == nullptr || hitProgram == nullptr || missProgram == nullptr)
    {
        printf("Could not create program groups for pipeline: (RayGenProgram: %p , HitProgram: %p, MissProgram: %p) \n",
            rayGenProgram,
            hitProgram,
            missProgram);
        return false;
    }

    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    pipelineLinkOptions.maxTraceDepth = 1;

    OptixProgramGroup programGroups[] = { rayGenProgram, missProgram, hitProgram };

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

    puts(log);

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

OptixModule OptixWrapper::CreateModule(const std::filesystem::path& a_PtxPath, const OptixPipelineCompileOptions& a_PipelineOptions) const
{

    OptixModuleCompileOptions moduleOptions = {};
    moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
    moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;

    std::ifstream stream;
    stream.open(a_PtxPath);

    if (!stream.is_open()){ return nullptr; }

    std::stringstream stringStream;
    stringStream << stream.rdbuf();
    std::string source = stringStream.str();

    char log[2048]; 
    auto logSize = sizeof(log);
    OptixModule module{};



    OptixResult error{};

    CHECKOPTIXRESULT(error = optixModuleCreateFromPTX(
        m_DeviceContext,
        &moduleOptions,
        &a_PipelineOptions,
        source.c_str(),
        source.size(),
        log,
        &logSize,
        &module));

    if (error)
    {
        puts(log);
        abort();
    }

    return module;

}

OptixProgramGroup OptixWrapper::CreateProgramGroup(OptixProgramGroupDesc a_ProgramGroupDesc, const std::string& a_Name)
{

    OptixProgramGroupOptions programGroupOptions = {};

    char log[2048];
    auto logSize = sizeof(log);

    OptixProgramGroup programGroup;



    OptixResult error{};

    CHECKOPTIXRESULT(error = optixProgramGroupCreate(
        m_DeviceContext,
        &a_ProgramGroupDesc,
        1,
        &programGroupOptions,
        log,
        &logSize,
        &programGroup));

    if (error)
    {
        puts(log);
        abort();
    }

    m_ProgramGroups.emplace(a_Name, programGroup);

    return programGroup;

}

void OptixWrapper::DestroyProgramGroups()
{

    for (const auto& programGroupRecord : m_ProgramGroups)
    {

        const auto programGroup = programGroupRecord.second;
        CHECKOPTIXRESULT(optixProgramGroupDestroy(programGroup));

    }

    m_ProgramGroups.clear();

}

void OptixWrapper::AccumulateStackSizes(OptixProgramGroup a_ProgramGroup, OptixStackSizes& a_StackSizes)
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

ProgramGroupHeader OptixWrapper::GetProgramGroupHeader(const std::string& a_GroupName) const
{

    assert(m_ProgramGroups.count(a_GroupName) != 0);

    ProgramGroupHeader header{};

    CHECKOPTIXRESULT(optixSbtRecordPackHeader(m_ProgramGroups.at(a_GroupName), &header));

    return header;

}

ComputedStackSizes OptixWrapper::ComputeStackSizes(
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

void OptixWrapper::SetupPipelineBuffer()
{

    m_OptixLaunchParamBuffer = std::make_unique<MemoryBuffer>(sizeof(WaveFront::OptixLaunchParameters));

}

void OptixWrapper::SetupShaderBindingTable()
{

    m_RayGenRecord = m_SBTGenerator->SetRayGen<void>();
    m_HitRecord = m_SBTGenerator->AddHitGroup<void>();
    m_MissRecord = m_SBTGenerator->AddMiss<void>();

    auto& rayGenRecord = m_RayGenRecord.GetRecord();
    rayGenRecord.m_Header = GetProgramGroupHeader(s_RayGenPGName);

    auto& hitRecord = m_HitRecord.GetRecord();
    hitRecord.m_Header = GetProgramGroupHeader(s_HitPGName);

    auto& raysMissRecord = m_MissRecord.GetRecord();
    raysMissRecord.m_Header = GetProgramGroupHeader(s_MissPGName);

}

void OptixWrapper::OptixDebugCallback(unsigned a_Level, const char* a_Tag, const char* a_Message, void*)
{

    std::printf("%u::%s:: %s\n\n", a_Level, a_Tag, a_Message);

}



void OptixWrapper::UpdateSBT()
{

    m_SBTGenerator->UpdateTable();

}

void OptixWrapper::TraceRays(
    unsigned int a_NumRays,
    const OptixLaunchParameters& a_LaunchParams,  
    CUstream a_CUDAStream) const
{

    m_OptixLaunchParamBuffer->Write(a_LaunchParams);

    OptixShaderBindingTable shaderBindingTable = m_SBTGenerator->GetTableDesc();

    CHECKOPTIXRESULT(optixLaunch(m_Pipeline,
        a_CUDAStream,
        m_OptixLaunchParamBuffer->GetCUDAPtr(),
        m_OptixLaunchParamBuffer->GetSize(),
        &shaderBindingTable,
        a_NumRays,
        1,
        1));

    cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;

}
#endif
