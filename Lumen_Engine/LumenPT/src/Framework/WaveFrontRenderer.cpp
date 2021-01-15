#include "WaveFrontRenderer.h"
#include "PTMesh.h"
#include "PTScene.h"
#include "AccelerationStructure.h"
#include "Material.h"
#include "Texture.h"
#include "MemoryBuffer.h"
#include "OutputBuffer.h"
#include "ShaderBindingTableGen.h"
#include "../Shaders/CppCommon/LumenPTConsts.h"
#include "../CUDAKernels/WaveFrontKernels.cuh"

#include <cstdio>
#include <fstream>
#include <sstream>
#include "Cuda/cuda.h"
#include "Cuda/cuda_runtime.h"
#include "Optix/optix_stubs.h"
#include <glm/gtx/compatibility.hpp>


#include "PTMesh.h"
#include "PTPrimitive.h"

void CheckOptixRes(const OptixResult& a_res)
{
    if(a_res != OPTIX_SUCCESS)
    {
        std::string errorName = optixGetErrorName(a_res);
        std::string errorMessage = optixGetErrorString(a_res);

        std::fprintf(
            stderr, 
            "Optix error occured: %s \n Description: %s", 
            errorName.c_str(), 
            errorMessage.c_str());
    }
}

#if defined(OPTIX_NOCHECK)
#define CHECKOPTIXRESULT(x)
#elif defined(OPTIX_CHECK) || defined(_DEBUG)
#define CHECKOPTIXRESULT(x)\
    CheckOptixRes(x);
#endif

WaveFrontRenderer::WaveFrontRenderer(const InitializationData& a_InitializationData)
    :
m_ServiceLocator({}),
m_DeviceContext(nullptr),
m_PipelineRays(nullptr),
m_PipelineShadowRays(nullptr),
m_ShaderBindingTableGenerator(nullptr),
m_ProgramGroups({}),
m_OutputBuffer(nullptr),
m_SBTBuffer(nullptr),
m_TempBuffers(),
m_PrimRaysPrevFrameBatchIndex(0u),
m_ResultBuffer(nullptr),
m_PixelBuffer3Channels(nullptr),
m_PixelBuffer1Channel(nullptr),
m_RayBatches(),
m_IntersectionBuffers(),
m_ShadowRayBatch(nullptr),
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

    m_ShaderBindingTableGenerator = std::make_unique<ShaderBindingTableGenerator>();

    m_Texture = std::make_unique<Texture>(LumenPTConsts::gs_AssetDirectory + "debugTex.jpg");

    m_ServiceLocator.m_SBTGenerator = m_ShaderBindingTableGenerator.get();
    m_ServiceLocator.m_Renderer = this;

    CreateShaderBindingTable();


}

WaveFrontRenderer::~WaveFrontRenderer()
{}

bool WaveFrontRenderer::Initialize(const InitializationData& a_InitializationData)
{
    bool success = true;\
    //Temporary(put into init data)
    const std::string shaderPath = LumenPTConsts::gs_ShaderPathBase + "WaveFrontShaders.ptx";

    m_Resolution = a_InitializationData.m_Resolution;
    m_MaxDepth = a_InitializationData.m_MaxDepth;

    InitializeContext();
    success &= CreatePipelines(shaderPath);
    CreateOutputBuffer();
    CreateDataBuffers();

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
    CheckOptixRes(optixAccelBuild(m_DeviceContext, 0, &a_BuildOptions, &a_BuildInput, 1, *tempBuffer, sizes.tempSizeInBytes,
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
void WaveFrontRenderer::CreateShaderBindingTable()
{
    

   
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

    //Allocate and initialize ray batches.
    for(auto& rayBatch : m_RayBatches)
    {
        rayBatch = std::make_unique<MemoryBuffer>(sizeof(RayBatch) + numPixels * raysPerPixel * sizeof(RayData));
        rayBatch->Write(numPixels, 0);
        rayBatch->Write(raysPerPixel, sizeof(RayBatch::m_NumPixels));
    }

    for(auto& intersectionBuffer : m_IntersectionBuffers)
    {
        intersectionBuffer = std::make_unique<MemoryBuffer>(sizeof(IntersectionBuffer) + numPixels * raysPerPixel * sizeof(IntersectionData));
        intersectionBuffer->Write(numPixels, 0);
        intersectionBuffer->Write(raysPerPixel, sizeof(IntersectionBuffer::m_NumPixels));
    }

    m_ShadowRayBatch = std::make_unique<MemoryBuffer>(sizeof(ShadowRayBatch) + maxDepth * numPixels * shadowRaysPerPixel * sizeof(ShadowRayData));
    m_ShadowRayBatch->Write(maxDepth, 0);
    m_ShadowRayBatch->Write(numPixels, sizeof(ShadowRayBatch::m_MaxDepth));
    m_ShadowRayBatch->Write(shadowRaysPerPixel, sizeof(ShadowRayBatch::m_MaxDepth) + sizeof(ShadowRayBatch::m_NumPixels));

}

void WaveFrontRenderer::DebugCallback(unsigned a_Level, const char* a_Tag, const char* a_Message, void*)
{

    std::printf("%u::%s:: %s\n\n", a_Level, a_Tag, a_Message);

}

void WaveFrontRenderer::AccumulateStackSizes(OptixProgramGroup a_ProgramGroup, OptixStackSizes& a_StackSizes)
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

ProgramGroupHeader WaveFrontRenderer::GetProgramGroupHeader(const std::string& a_GroupName) const
{

    assert(m_ProgramGroups.count(a_GroupName) != 0);

    ProgramGroupHeader header{};

    optixSbtRecordPackHeader(m_ProgramGroups.at(a_GroupName), &header);

    return header;

}

GLuint WaveFrontRenderer::TraceFrame()
{

    //Generate Camera rays using CUDA kernel.
    float3 eye, u, v, w;
    m_Camera.GetVectorData(eye, u, v, w);

    const WaveFront::DeviceCameraData cameraData(eye, u, v, w);
    const unsigned int newPrimRayBatchIndex = GetNextPrimRayBatchIndex(false);
    MemoryBuffer& newPrimaryRayBatch = *m_RayBatches[newPrimRayBatchIndex];
    
    const WaveFront::SetupLaunchParameters setupParams(m_Resolution, cameraData, newPrimaryRayBatch.GetDevicePtr<RayBatch>());
    //GenerateRays(setupParams);

    //Loop
    //Trace buffer of rays using Optix ResolveRays pipeline
    //Calculate shading for intersections in buffer using CUDA kernel.

    for(unsigned waveIndex = 0; waveIndex < m_MaxDepth; ++waveIndex)
    {

        //Launch Optix ResolveRays pipeline

        const uint3 resolutionAndDepth = make_uint3(m_Resolution.x, m_Resolution.y, waveIndex);

        const unsigned int oldPrimRayBatchIndex = m_PrimRaysPrevFrameBatchIndex;
        MemoryBuffer& oldPrimRayBatch = *m_RayBatches[oldPrimRayBatchIndex];
        

    }

    //Trace buffer of shadow rays using Optix ResolveShadowRays.
    //Post processing using CUDA kernel.

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

unsigned WaveFrontRenderer::GetNextRayBatchIndex(unsigned a_CurrentDepth)
{
    //Never return PrimaryRaysPrevFrame batch index.
    //To make it easier to select a batch index that is not the PrimaryRaysPrevFrame batch, use only the first and last indices for the primary ray batches.
    //This makes its easier because you either include the first index (0) when the PrimaryRaysPrevFrame batch is not 0 and run up to lastIndex -1;
    //or when the PrimaryRaysPrevFrame buffer is 0 run, start from index 1 and run up to lastIndex.
    //This means that the range is always: (num ray batches - 1)

    const unsigned lastIndex = s_NumRayBatches - 1;
    const bool usesLastBatch = (m_PrimRaysPrevFrameBatchIndex == (lastIndex));

    unsigned index = a_CurrentDepth % (lastIndex); //This number includes the lastIndex but since it returns remainder it will never be lastIndex.
    //Example: lastIndex = 2, 0 % 2 = 0, 1 % 2 = 1, 2 % 2 = 0. This makes the range to be: (num ray batches -1)

    if (usesLastBatch) //If the primary ray batch is on the last index
    {
        return index; //range = 0 -> lastIndex-1
    }
    else //the primary ray batch is on the first index
    {
        return index + 1; //range = 1 -> lastIndex
    }

}

unsigned WaveFrontRenderer::GetNextPrimRayBatchIndex(bool a_Overwrite)
{

    const unsigned lastIndex = s_NumRayBatches - 1;
    const bool usesLastBatch = (m_PrimRaysPrevFrameBatchIndex == (lastIndex));

    if(usesLastBatch)
    {
        if (a_Overwrite)
        {
            m_PrimRaysPrevFrameBatchIndex = 0;
        }

        return 0;
    }
    else
    {
        if(a_Overwrite)
        {
            m_PrimRaysPrevFrameBatchIndex = lastIndex;
        }

        return lastIndex;
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

    prim->m_RecordHandle = m_ShaderBindingTableGenerator->AddHitGroup<DevicePrimitive>();
    auto& rec = prim->m_RecordHandle.GetRecord();
    rec.m_Header = GetProgramGroupHeader("Hit");
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
