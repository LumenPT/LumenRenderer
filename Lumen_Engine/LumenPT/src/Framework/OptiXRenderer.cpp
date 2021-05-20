
#if ! defined(WAVEFRONT)
#include "OptiXRenderer.h"

#include "PTMesh.h"
#include "PTScene.h"
#include "AccelerationStructure.h"
#include "PTMaterial.h"
#include "PTTexture.h"
#include "MemoryBuffer.h"
#include "CudaGLTexture.h"
#include "ShaderBindingTableGen.h"
#include "../Shaders/CppCommon/SceneDataTableAccessor.h"
#include "CudaUtilities.h"
#include "SceneDataTable.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <filesystem>
#include "Cuda/cuda.h"
#include "Cuda/cuda_runtime.h"
#include "Optix/optix_stubs.h"
#include <glm/gtx/compatibility.hpp>
#include <Optix/optix_function_table_definition.h>

#include "PTVolume.h"

const uint32_t gs_ImageWidth = 800;
const uint32_t gs_ImageHeight = 600;


#include "PTMesh.h"
#include "PTPrimitive.h"
#include "PTVolume.h"

OptiXRenderer::OptiXRenderer(const InitializationData& a_InitializationData)
    :
    m_DeviceContext(nullptr),
    m_PipelineCompileOptions({}),
    m_Pipeline(nullptr),
    m_RayGenRecord({}),
    m_MissRecord({}),
    m_HitRecord({}),
    m_Texture(nullptr),
    m_ProgramGroups({}),
    m_OutputBuffer(nullptr),
    m_SBTBuffer(nullptr),
    m_InitData(a_InitializationData)
{
    bool success = true;

    InitializeContext();
    InitializePipelineOptions();
    CreatePipeline();
    CreateOutputBuffer();

    m_ShaderBindingTableGenerator = std::make_unique<ShaderBindingTableGenerator>();

    m_ServiceLocator.m_Renderer = this;
    m_Texture = std::make_unique<PTTexture>(m_InitData.m_AssetDirectory.string() + "debugTex.jpg");

    CreateShaderBindingTable();

    m_SceneDataTable = std::make_unique<SceneDataTable>();
    m_ServiceLocator.m_SceneDataTable = m_SceneDataTable.get();

    // m_Scene->m_Camera->SetPosition(glm::vec3(0.f, 0.f, -50.f));
}

OptiXRenderer::~OptiXRenderer()
{}

void OptiXRenderer::InitializeContext()
{

    cudaFree(0);
    CUcontext cu_ctx = 0;
    CHECKOPTIXRESULT(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &OptiXRenderer::DebugCallback;
    options.logCallbackLevel = 4;
    CHECKOPTIXRESULT(optixDeviceContextCreate(cu_ctx, &options, &m_DeviceContext));

}

void OptiXRenderer::InitializePipelineOptions()
{

    m_PipelineCompileOptions = {};
    m_PipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    m_PipelineCompileOptions.numAttributeValues = 3;
    // Defines how many 32-bit values can be output by a miss or hit shader
    m_PipelineCompileOptions.numPayloadValues = 5;
    // What exceptions can the pipeline throw.
    m_PipelineCompileOptions.exceptionFlags = OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_DEBUG
        | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    // 0 corresponds to enabling custom primitives and triangles, but nothing else 
    m_PipelineCompileOptions.usesPrimitiveTypeFlags = 0;
    m_PipelineCompileOptions.usesMotionBlur = false;
    m_PipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

}

//TODO: implement necessary Optix shaders for wavefront algorithm
//Shaders:
// - RayGen: trace rays from ray definitions in ray batch, different methods for radiance or shadow.
// - Miss: is this shader necessary, miss result by default in intersection buffer?
// - ClosestHit: radiance trace, report intersection into intersection buffer.
// - AnyHit: shadow trace, report if any hit in between certain max distance.
void OptiXRenderer::CreatePipeline()
{

    //OptixModule shaderModule = CreateModule(LumenPTConsts::gs_ShaderPathBase + "WaveFrontShaders.ptx");
    OptixModule shaderModule = CreateModule( + "draw_solid_color.ptx");

    OptixProgramGroupDesc rgGroupDesc = {};
    rgGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    //rgGroupDesc.raygen.entryFunctionName = "__raygen__WaveFrontGenRays";
    rgGroupDesc.raygen.entryFunctionName = "__raygen__draw_solid_color";
    rgGroupDesc.raygen.module = shaderModule;

    OptixProgramGroup rayGenProgram = CreateProgramGroup(rgGroupDesc, "RayGen");

    OptixProgramGroupDesc msGroupDesc = {};
    msGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    //msGroupDesc.miss.entryFunctionName = "__miss__WaveFrontMiss";
    msGroupDesc.miss.entryFunctionName = "__miss__MissShader";
    msGroupDesc.miss.module = shaderModule;

    OptixProgramGroup missProgram = CreateProgramGroup(msGroupDesc, "Miss");

    OptixProgramGroupDesc htGroupDesc = {};
    htGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    //htGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__WaveFrontClosestHit";
    htGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__HitShader";
    //htGroupDesc.hitgroup.entryFunctionNameAH = "__anyhit__WaveFrontAnyHit";
    htGroupDesc.hitgroup.moduleCH = shaderModule;
    //htGroupDesc.hitgroup.moduleAH = shaderModule;

    OptixProgramGroup hitProgram = CreateProgramGroup(htGroupDesc, "Hit");

    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    pipelineLinkOptions.maxTraceDepth = 3; //TODO: potentially add this as a init param.

    //volumetric_bookmark
    OptixModule volumetricShaderModule = CreateModule(m_InitData.m_ShaderDirectory.string() + "volumetric.ptx");

    assert(volumetricShaderModule);

    OptixProgramGroupDesc volumetric_htGroupdesc = {};
    volumetric_htGroupdesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    volumetric_htGroupdesc.hitgroup.moduleCH = volumetricShaderModule;
    volumetric_htGroupdesc.hitgroup.entryFunctionNameCH = "__closesthit__VolumetricHitShader";
    volumetric_htGroupdesc.hitgroup.moduleAH = volumetricShaderModule;
    volumetric_htGroupdesc.hitgroup.entryFunctionNameAH = "__anyhit__VolumetricHitShader";
    volumetric_htGroupdesc.hitgroup.moduleIS = volumetricShaderModule;
    volumetric_htGroupdesc.hitgroup.entryFunctionNameIS = "__intersection__VolumetricHitShader";

    OptixProgramGroup volumetrichitProgram = CreateProgramGroup(volumetric_htGroupdesc, "VolumetricHit");

    //volumetric_bookmark
    OptixProgramGroup programGroups[] = { rayGenProgram, missProgram, hitProgram, volumetrichitProgram };

    char log[2048];
    auto logSize = sizeof(log);

    OptixResult error{};

    CHECKOPTIXRESULT(error = optixPipelineCreate(
        m_DeviceContext,
        &m_PipelineCompileOptions,
        &pipelineLinkOptions,
        programGroups,
        sizeof(programGroups) / sizeof(OptixProgramGroup),
        log,
        &logSize,
        &m_Pipeline));

    OptixStackSizes stackSizes = {};

    AccumulateStackSizes(rayGenProgram, stackSizes);
    AccumulateStackSizes(missProgram, stackSizes);
    AccumulateStackSizes(hitProgram, stackSizes);

    auto finalSizes = ComputeStackSizes(stackSizes, 1, 0, 0);

    CHECKOPTIXRESULT(error = optixPipelineSetStackSize(
        m_Pipeline,
        finalSizes.DirectSizeTrace,
        finalSizes.DirectSizeState,
        finalSizes.ContinuationSize,
        3));
}

OptixModule OptiXRenderer::CreateModule(const std::string& a_PtxPath)
{
    assert(std::filesystem::exists(a_PtxPath));

    OptixModuleCompileOptions moduleOptions = {};
    moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
    moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;

    std::ifstream stream;
    stream.open(a_PtxPath);



    std::stringstream stringStream;
    stringStream << stream.rdbuf();
    std::string source = stringStream.str();

    char log[2048];
    auto logSize = sizeof(log);
    OptixModule module;

    CHECKOPTIXRESULT(optixModuleCreateFromPTX(
        m_DeviceContext,
        &moduleOptions,
        &m_PipelineCompileOptions,
        source.c_str(),
        source.size(),
        log,
        &logSize,
        &module));

    return module;

}

OptixProgramGroup OptiXRenderer::CreateProgramGroup(OptixProgramGroupDesc a_ProgramGroupDesc, const std::string& a_Name)
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

std::unique_ptr<AccelerationStructure> OptiXRenderer::BuildGeometryAccelerationStructure(
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

std::unique_ptr<AccelerationStructure> OptiXRenderer::BuildInstanceAccelerationStructure(std::vector<OptixInstance> a_Instances)
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

//TODO: adjust output buffer to the necessary output buffer received
//from CUDA.
void OptiXRenderer::CreateOutputBuffer()
{

    m_OutputBuffer = std::make_unique<::CudaGLTexture>(gs_ImageWidth, gs_ImageHeight);

}

void OptiXRenderer::DebugCallback(unsigned a_Level, const char* a_Tag, const char* a_Message, void*)
{

    std::printf("%u::%s:: %s\n\n", a_Level, a_Tag, a_Message);

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

//TODO: replace data structures with the data structures necessary
//in this rendering pipeline. (Wavefront data structures)
void OptiXRenderer::CreateShaderBindingTable()
{

    m_RayGenRecord = m_ShaderBindingTableGenerator->SetRayGen<RaygenData>();

    auto& rayGenRecord = m_RayGenRecord.GetRecord();
    rayGenRecord.m_Header = GetProgramGroupHeader("RayGen");
    rayGenRecord.m_Data.m_Color = { 0.4f, 0.5f, 0.2f };

    m_MissRecord = m_ShaderBindingTableGenerator->AddMiss<MissData>();

    auto& missRecord = m_MissRecord.GetRecord();
    missRecord.m_Header = GetProgramGroupHeader("Miss");
    missRecord.m_Data.m_Num = 2;
    missRecord.m_Data.m_Color = { 0.f, 0.f, 1.f };

    //bookmark
    m_HitRecord = m_ShaderBindingTableGenerator->AddHitGroup<void>();

    auto& hitRecord = m_HitRecord.GetRecord();
    hitRecord.m_Header = GetProgramGroupHeader("Hit");
}

ProgramGroupHeader OptiXRenderer::GetProgramGroupHeader(const std::string& a_GroupName) const
{

    assert(m_ProgramGroups.count(a_GroupName) != 0);

    ProgramGroupHeader header;

    optixSbtRecordPackHeader(m_ProgramGroups.at(a_GroupName), &header);

    return header;

}

unsigned int OptiXRenderer::TraceFrame(std::shared_ptr<Lumen::ILumenScene>& a_Scene)
{
    std::vector<float3> vert = {
        {0.5f, 0.5f, 0.5f},
        {0.5f, -0.5f, 0.5f},
        {-0.5f, 0.5f, 0.5f}
    };
    std::vector<Vertex> verti = std::vector<Vertex>(3);
    verti[0].m_Position = vert[0];
    verti[0].m_Normal = { 1.0f, 0.0f, 0.0f };
    verti[1].m_Position = vert[1];
    verti[1].m_Normal = { 0.0f, 1.0f, 0.0f };
    verti[2].m_Position = vert[2];
    verti[2].m_Normal = { 0.0f, 0.0f, 1.0f };

    MemoryBuffer vertexBuffer(verti.size() * sizeof(Vertex));
    vertexBuffer.Write(verti.data(), verti.size() * sizeof(Vertex), 0);

    LaunchParameters params = {};
    OptixShaderBindingTable sbt = m_ShaderBindingTableGenerator->GetTableDesc();

    params.m_SceneData = m_SceneDataTable->GetDevicePointer();
    auto str = static_cast<PTScene*>(a_Scene.get())->GetSceneAccelerationStructure();

    params.m_Image = m_OutputBuffer->GetDevicePtr();
    params.m_Handle = str;

    params.m_ImageWidth = gs_ImageWidth;
    params.m_ImageHeight = gs_ImageHeight;
    params.m_VertexBuffer = vertexBuffer.GetDevicePtr<Vertex>();

    a_Scene->m_Camera->SetAspectRatio(static_cast<float>(gs_ImageWidth) / static_cast<float>(gs_ImageHeight));
    glm::vec3 eye, U, V, W;
    a_Scene->m_Camera->GetVectorData(eye, U, V, W);
    params.eye = make_float3(eye.x, eye.y, eye.z);
    params.U = make_float3(U.x, U.y, U.z);
    params.V = make_float3(V.x, V.y, V.z);
    params.W = make_float3(W.x, W.y, W.z);

    MemoryBuffer devBuffer(sizeof(params));
    devBuffer.Write(params);

    auto res = optixLaunch(
        m_Pipeline,
        0,
        *devBuffer,
        devBuffer.GetSize(),
        &sbt,
        gs_ImageWidth,
        gs_ImageHeight,
        1);

    return m_OutputBuffer->GetTexture();

}

OptiXRenderer::ComputedStackSizes OptiXRenderer::ComputeStackSizes(
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

std::unique_ptr<MemoryBuffer> OptiXRenderer::InterleaveVertexData(const PrimitiveData& a_MeshData)
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

std::shared_ptr<Lumen::ILumenTexture> OptiXRenderer::CreateTexture(void* a_PixelData, uint32_t a_Width, uint32_t a_Height)
{

    static cudaChannelFormatDesc formatDesc = cudaCreateChannelDesc<uchar4>();
    return std::make_shared<PTTexture>(a_PixelData, formatDesc, a_Width, a_Height);

}

std::unique_ptr<Lumen::ILumenPrimitive> OptiXRenderer::CreatePrimitive(PrimitiveData& a_PrimitiveData)
{
    auto vertexBuffer = InterleaveVertexData(a_PrimitiveData);

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

    prim->m_SceneDataTableEntry = m_SceneDataTable->AddEntry<DevicePrimitive>();
    auto& entry = prim->m_SceneDataTableEntry.GetData();
    entry.m_VertexBuffer = prim->m_VertBuffer->GetDevicePtr<Vertex>();
    entry.m_IndexBuffer = prim->m_IndexBuffer->GetDevicePtr<unsigned int>();
    entry.m_Material = static_cast<PTMaterial*>(prim->m_Material.get())->GetDeviceMaterial();

    return prim;
}

std::shared_ptr<Lumen::ILumenMesh> OptiXRenderer::CreateMesh(
    std::vector<std::shared_ptr<Lumen::ILumenPrimitive>>& a_Primitives)
{
    auto mesh = std::make_shared<PTMesh>(a_Primitives, m_ServiceLocator);
    return mesh;
}

std::shared_ptr<Lumen::ILumenMaterial> OptiXRenderer::CreateMaterial(const MaterialData& a_MaterialData)
{

    auto mat = std::make_shared<PTMaterial>();
    mat->SetDiffuseColor(a_MaterialData.m_DiffuseColor);
    mat->SetDiffuseTexture(a_MaterialData.m_DiffuseTexture);

    return mat;

}

std::shared_ptr<Lumen::ILumenScene> OptiXRenderer::CreateScene(SceneData a_SceneData)
{
    return std::make_shared<PTScene>(a_SceneData, m_ServiceLocator);
}

std::shared_ptr<Lumen::ILumenVolume> OptiXRenderer::CreateVolume(const std::string& a_FilePath)
{
    std::shared_ptr<PTVolume> volume = std::make_shared<PTVolume>(a_FilePath, m_ServiceLocator);

    //volumetric_bookmark
    volume->m_RecordHandle = m_ShaderBindingTableGenerator->AddHitGroup<DeviceVolume>();
    auto& rec = volume->m_RecordHandle.GetRecord();
    rec.m_Header = GetProgramGroupHeader("VolumetricHit");
    rec.m_Data.m_Grid = volume->m_Handle.grid<float>();

    volume->m_SceneEntry = m_SceneDataTable->AddEntry<DeviceVolume>();
    volume->m_SceneEntry.GetData().m_Grid = volume->m_Handle.grid<float>();

    uint32_t geomFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE };

    OptixAccelBuildOptions buildOptions = {};
    buildOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
    buildOptions.motionOptions = {};

    OptixAabb aabb = { -1.5f, -1.5f, -1.5f, 1.5f, 1.5f, 1.5f };

    auto grid = volume->GetHandle()->grid<float>();
    auto bbox = grid->worldBBox();

    nanovdb::Vec3<double> temp = bbox.min();
    float bboxMinX = bbox.min()[0];
    float bboxMinY = bbox.min()[1];
    float bboxMinZ = bbox.min()[2];
    float bboxMaxX = bbox.max()[0];
    float bboxMaxY = bbox.max()[1];
    float bboxMaxZ = bbox.max()[2];

    aabb = { bboxMinX, bboxMinY, bboxMinZ, bboxMaxX, bboxMaxY, bboxMaxZ };

    MemoryBuffer aabb_buffer(sizeof(OptixAabb));
    aabb_buffer.Write(aabb);

    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    buildInput.customPrimitiveArray.aabbBuffers = &*aabb_buffer;
    buildInput.customPrimitiveArray.numPrimitives = 1;
    buildInput.customPrimitiveArray.flags = geomFlags;
    buildInput.customPrimitiveArray.numSbtRecords = 1;

    volume->m_AccelerationStructure = BuildGeometryAccelerationStructure(buildOptions, buildInput);
    m_testVolumeGAS = volume->m_AccelerationStructure.get();


    return volume;
}

#endif
