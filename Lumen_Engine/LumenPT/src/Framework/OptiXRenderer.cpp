#include "OptiXRenderer.h"
#include "../Shaders/CppCommon/LumenPTConsts.h"

#include "MemoryBuffer.h"
#include "OutputBuffer.h"

#include "../Shaders/CppCommon/LaunchParameters.h"

#include "Optix/optix_stubs.h"

#include "Cuda/cuda.h"
#include "Cuda/cuda_gl_interop.h"
#include "Cuda/cuda_runtime.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cassert>

#include <bitset>
#include <iostream>


OptiXRenderer::OptiXRenderer(const InitializationData& /*a_InitializationData*/)
{
    InitializeContext();
    InitializePipelineOptions();
    CreatePipeline();
    CreateOutputBuffer();

    cuStreamCreate(&m_CudaStream, CU_STREAM_DEFAULT);
}

OptiXRenderer::~OptiXRenderer()
{
    // To move the destructor from the header to this file
}

void OptiXRenderer::InitializeContext()
{
    auto str = LumenPTConsts::gs_ShaderPathBase;

    cudaFree(0);
    CUcontext          cu_ctx = 0;  // zero means take the current context
    optixInit();
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &OptiXRenderer::DebugCallback;
    options.logCallbackLevel = 4; // Determine what messages are sent out via the callback function. 4 is the most verbose, 0 is silent.
    optixDeviceContextCreate(cu_ctx, &options, &m_DeviceContext);
}

void OptiXRenderer::InitializePipelineOptions()
{
    m_PipelineCompileOptions.traversableGraphFlags = OptixTraversableGraphFlags::OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    m_PipelineCompileOptions.numAttributeValues = 2; // Defaults to 2, maximum is 8. Defines how many 32-bit values can be output from an intersection shader
    m_PipelineCompileOptions.numPayloadValues = 3; // Defines how many 32-bit values can be output by a hit shader
    m_PipelineCompileOptions.exceptionFlags = OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_DEBUG;
    m_PipelineCompileOptions.usesPrimitiveTypeFlags = 0; // 0 corresponds to enabling custom primitives and triangles, but nothing else 
    m_PipelineCompileOptions.usesMotionBlur = false;
    // Name by which the launch parameters will be accessible in the shaders. This must match with the shaders.
    m_PipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

}

void OptiXRenderer::CreatePipeline()
{	
    // TODO: This needs to be modified when we have the actual shaders for the pipeline
    // Also might need multiple pipelines in the entire renderer because of the wavefront shenanigans
    // In that case this function will be made modular with more arguments defining the pipeline

    // Create a module, these need to be saved somehow I believe
    // Note multiple program groups can be created from a single module
    auto someMod = CreateModule(LumenPTConsts::gs_ShaderPathBase + "draw_solid_color.ptx");

    // Descriptor of the program group, including its type, modules it uses and the name of the entry function
    OptixProgramGroupDesc rtGroupDesc = {};
    rtGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rtGroupDesc.raygen.entryFunctionName = "__raygen__draw_solid_color";
    rtGroupDesc.raygen.module = someMod;

    auto raygen = CreateProgramGroup(rtGroupDesc, "RayGen");

    OptixProgramGroupDesc msGroupDesc = {};
    msGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    msGroupDesc.miss.entryFunctionName = "__miss__MissShader";
    msGroupDesc.miss.module = someMod;

    auto miss = CreateProgramGroup(msGroupDesc, "Miss");
    OptixProgramGroupDesc htGroupDesc = {};
    htGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    htGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__HitShader";
    htGroupDesc.hitgroup.moduleCH = someMod;

    auto hit = CreateProgramGroup(htGroupDesc, "Hit");


    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
    pipelineLinkOptions.maxTraceDepth = 3; // 2 or 3, needs some profiling

    // The program groups to include in the pipeline, can be replaced with an std::vector
    OptixProgramGroup programGroups[] = { raygen, miss, hit};

    char log[2048];
    auto logSize = sizeof(log);

    optixPipelineCreate(m_DeviceContext, &m_PipelineCompileOptions, &pipelineLinkOptions, programGroups,
        sizeof(programGroups) / sizeof(OptixProgramGroup), log, &logSize, &m_Pipeline);

    OptixStackSizes stackSizes = {};

    AccumulateStackSizes(raygen, stackSizes);/*
    AccumulateStackSizes(miss, stackSizes);
    AccumulateStackSizes(hit, stackSizes);*/

    auto finalSizes = ComputeStackSizes(stackSizes, 1, 0, 0);

    optixPipelineSetStackSize(m_Pipeline, finalSizes.DirectSizeTrace,
        finalSizes.DirectSizeState, finalSizes.ContinuationSize, 2);

}

OptixModule OptiXRenderer::CreateModule(const std::string& a_PtxPath)
{
    OptixModuleCompileOptions moduleOptions = {};
    moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
    moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;


    // Open the ptx file as an input stream and load all of it into a string using a string stream
    std::ifstream stream;
    stream.open(a_PtxPath);
    std::stringstream stringStream;
    stringStream << stream.rdbuf();
    std::string source = stringStream.str();

    char log[2048];
    auto logSize = sizeof(log);
    OptixModule mod;

    optixModuleCreateFromPTX(m_DeviceContext, &moduleOptions, &m_PipelineCompileOptions, source.c_str(), source.size(), log, &logSize, &mod);

    return mod;
}

OptixProgramGroup OptiXRenderer::CreateProgramGroup(OptixProgramGroupDesc a_ProgramGroupDesc, const std::string& a_Name)
{
    OptixProgramGroupOptions programGroupOptions = {}; // Placeholder structure, so it's empty

    char log[2048];
    auto logSize = sizeof(log);

    OptixProgramGroup programGroup;

    optixProgramGroupCreate(m_DeviceContext, &a_ProgramGroupDesc, 1, &programGroupOptions, log, &logSize, &programGroup);

    m_ProgramGroups.emplace(a_Name, programGroup);

    return programGroup;
}

OptixTraversableHandle OptiXRenderer::BuildGeometryAccelerationStructure(const OptixAccelBuildOptions& a_BuildOptions,
    const OptixBuildInput& a_BuildInput)
{
    
    // Let Optix compute how much memory the output buffer and the temporary buffer need to have, then create these buffers
    OptixAccelBufferSizes sizes;
    optixAccelComputeMemoryUsage(m_DeviceContext, &a_BuildOptions, &a_BuildInput, 1, &sizes);
    MemoryBuffer tempBuffer(sizes.tempSizeInBytes);
    MemoryBuffer resultBuffer(sizes.outputSizeInBytes);

    OptixTraversableHandle handle;
    // It is possible to have functionality similar to DX12 Command lists with cuda streams, though it is not required.
    // Just worth mentioning if we want that functionality.
    optixAccelBuild(m_DeviceContext, 0, &a_BuildOptions, &a_BuildInput, 1, *tempBuffer, sizes.tempSizeInBytes,
        *resultBuffer, sizes.outputSizeInBytes, &handle, nullptr, 0);

    // Needs to return the resulting memory buffer and the traversable handle
    return handle;
}

OptixTraversableHandle OptiXRenderer::BuildInstanceAccelerationStructure(std::vector<OptixInstance> a_Instances)
{
    // The fucky part here is figuring out how the build the a_Instances vector outside this function
    // since it will contain SBT indices, transforms and whatnot. Those will probably come from the static mesh instances we have in the scene


    MemoryBuffer instanceBuffer(a_Instances.size() * sizeof(OptixInstance));
    instanceBuffer.Write(a_Instances.data(), a_Instances.size() * sizeof(OptixInstance), 0);

    OptixBuildInput buildInput = {};
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
    optixAccelComputeMemoryUsage(m_DeviceContext, &buildOptions, &buildInput, 1, &sizes);

    MemoryBuffer tempBuffer(sizes.tempSizeInBytes);
    MemoryBuffer resultBuffer(sizes.outputSizeInBytes);

    OptixTraversableHandle handle;
    optixAccelBuild(m_DeviceContext, 0, &buildOptions, &buildInput, 1, *tempBuffer, sizes.tempSizeInBytes,
        *resultBuffer, sizes.outputSizeInBytes, &handle, nullptr, 0);

    // Needs to return the handle and the result buffer
    return handle;
}

void OptiXRenderer::CreateOutputBuffer()
{
    m_OutputBuffer = std::make_unique<OutputBuffer>(128, 128);
}

void OptiXRenderer::DebugCallback(unsigned int a_Level, const char* a_Tag, const char* a_Message, void*)
{
    std::printf("%u::%s:: %s\n\n", a_Level, a_Tag, a_Message); // This can be changed to better suite our preferred debug output format
}

void OptiXRenderer::AccumulateStackSizes(OptixProgramGroup a_ProgramGroup, OptixStackSizes& a_StackSizes)
{
    OptixStackSizes localSizes;
    optixProgramGroupGetStackSize(a_ProgramGroup, &localSizes);
    a_StackSizes.cssRG = std::max(a_StackSizes.cssRG, localSizes.cssRG);
    a_StackSizes.cssMS = std::max(a_StackSizes.cssMS, localSizes.cssMS);
    a_StackSizes.cssIS = std::max(a_StackSizes.cssIS, localSizes.cssIS);
    a_StackSizes.cssAH = std::max(a_StackSizes.cssAH, localSizes.cssAH);
    a_StackSizes.cssCH = std::max(a_StackSizes.cssCH, localSizes.cssCH);
    a_StackSizes.cssCC = std::max(a_StackSizes.cssCC, localSizes.cssCC);
    a_StackSizes.dssDC = std::max(a_StackSizes.dssDC, localSizes.dssDC);
}

OptixShaderBindingTable OptiXRenderer::CreateShaderBindingTable()
{
    // OptixShaderBindingTable is just a struct which describes where the different parts of the table are located in GPU memory
    // This means that the entire table could easily be split into multiple buffers if we so desire


    m_SBTBuffer = std::make_unique<MemoryBuffer>(256);

    struct RayGenData
    {
        float3 m_ClearColor;
    };

    SBTRecord<RayGenData> rayGenRecord;

    rayGenRecord.m_Header = GetProgramGroupHeader("RayGen");
    rayGenRecord.m_Data.m_ClearColor = { 0.4f, 0.5f, 0.9f };

    m_SBTBuffer->Write(rayGenRecord, 0);

    //struct MissData
    //{
    //    float3 m_Color;
    //};


    struct Empty{};
    SBTRecord<Empty> missRecord;

    missRecord.m_Header = GetProgramGroupHeader("Miss");

    auto missOffset = (sizeof(rayGenRecord) / 32 + 1) * 32;
    m_SBTBuffer->Write(missRecord, missOffset);

    SBTRecord<Empty> hitRecord;

    hitRecord.m_Header = GetProgramGroupHeader("Hit");

    auto hitOffset = missOffset + (sizeof(missRecord) / 32 + 1) * 32;

    m_SBTBuffer->Write(hitRecord, hitOffset);

    

    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord = **m_SBTBuffer;
    sbt.missRecordBase = **m_SBTBuffer + missOffset;
    sbt.missRecordCount = 1;
    sbt.missRecordStrideInBytes = (sizeof(missRecord) / OPTIX_SBT_RECORD_ALIGNMENT + 1) * OPTIX_SBT_RECORD_ALIGNMENT;
    sbt.hitgroupRecordBase = **m_SBTBuffer + hitOffset;
    sbt.hitgroupRecordCount = 1;
    sbt.hitgroupRecordStrideInBytes = (sizeof(hitRecord) / OPTIX_SBT_RECORD_ALIGNMENT + 1) * OPTIX_SBT_RECORD_ALIGNMENT;

    return sbt;
}

ProgramGroupHeader OptiXRenderer::GetProgramGroupHeader(const std::string& a_GroupName) const
{
    assert(m_ProgramGroups.count(a_GroupName) != 0);

    ProgramGroupHeader header;

    optixSbtRecordPackHeader(m_ProgramGroups.at(a_GroupName), &header);

    return header;
}

GLuint OptiXRenderer::TraceFrame()
{
    std::vector<float3> vert = {
        {0.5f, 0.5f, 0.5f},
        {-0.5f, 0.5f, 0.5f},
        {0.0f, 0.0f, 0.5f}
    };

    LaunchParameters params = {};

    params.m_Image = m_OutputBuffer->GetDevicePointer();
    params.m_Handle = BuildGeometryAccelerationStructure(vert);
    params.m_ImageWidth = 128;
    params.m_ImageHeight = 128;
    // Fill out struct here with whatev

    MemoryBuffer devBuffer(sizeof(params));
    devBuffer.Write(params);

    OptixShaderBindingTable sbt = CreateShaderBindingTable();


    optixLaunch(m_Pipeline, m_CudaStream, *devBuffer, devBuffer.GetSize(), &sbt, params.m_ImageWidth, params.m_ImageHeight, 1);
    cuStreamSynchronize(m_CudaStream);


    return m_OutputBuffer->GetTexture();
}

OptiXRenderer::ComputedStackSizes OptiXRenderer::ComputeStackSizes(OptixStackSizes a_StackSizes, int a_TraceDepth, int a_DirectDepth, int a_ContinuationDepth)
{
    ComputedStackSizes sizes;
    sizes.DirectSizeState = a_StackSizes.dssDC * a_DirectDepth;
    sizes.DirectSizeTrace = a_StackSizes.dssDC * a_DirectDepth;

    // upper bound on continuation stack used by call trees of continuation callables
    unsigned int cssCCTree = a_ContinuationDepth * a_StackSizes.cssCC;

    // upper bound on continuation stack used by CH or MS programs including the call tree of
    // continuation callables
    unsigned int cssCHOrMSPlusCCTree = std::max(a_StackSizes.cssCH, a_StackSizes.cssMS) + cssCCTree;

    // clang-format off
    sizes.ContinuationSize = a_StackSizes.cssRG + cssCCTree
        + (std::max(a_TraceDepth, 1) - 1) * cssCHOrMSPlusCCTree
        + std::min(a_TraceDepth, 1) * std::max(cssCHOrMSPlusCCTree, a_StackSizes.cssIS + a_StackSizes.cssAH);

    return sizes;
}

