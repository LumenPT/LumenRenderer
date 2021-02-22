#include "OptiXRenderer.h"
#include "../Shaders/CppCommon/LumenPTConsts.h"

#include "MemoryBuffer.h"
#include "OutputBuffer.h"
#include "AccelerationStructure.h"

#include "ShaderBindingTableGen.h"

#include "../Shaders/CppCommon/LaunchParameters.h"
#include "../Shaders/CppCommon/ModelStructs.h"

#include "PTMesh.h"

#include "PTPrimitive.h"
#include "Texture.h"

#include "Optix/optix_stubs.h"

#include "Cuda/cuda.h"
#include "Cuda/cuda_runtime.h"

#include "glm/glm.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cassert>

#include <bitset>
#include <iostream>

#include "Material.h"
#include "PTScene.h"
#include "PTVolume.h"

const uint32_t gs_ImageWidth = 800;
const uint32_t gs_ImageHeight = 600;

OptiXRenderer::OptiXRenderer(const InitializationData& /*a_InitializationData*/)
{
    InitializeContext();
    InitializePipelineOptions();
    CreatePipeline();
    CreateOutputBuffer();

    m_ShaderBindingTableGenerator = std::make_unique<ShaderBindingTableGenerator>();

    //m_ServiceLocator.m_Renderer = this;
    m_ServiceLocator.m_SBTGenerator = m_ShaderBindingTableGenerator.get();

    uchar4 px[] = {
        {255, 0, 0, 255 },
        {255, 255, 0, 255 },
        {255, 0, 0, 255 },
        {255, 255, 0, 255 }
    };
    //m_Texture = std::make_unique<Texture>(px, cudaCreateChannelDesc<uchar4>(), 2, 2);

    m_Texture = std::make_unique<Texture>(LumenPTConsts::gs_AssetDirectory + "debugTex.jpg");

    CreateShaderBindingTable();
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
    // Defaults to 2, maximum is 8. Defines how many 32-bit values can be output from an intersection shader
    m_PipelineCompileOptions.numAttributeValues = 2;
    // Defines how many 32-bit values can be output by a miss or hit shader
    m_PipelineCompileOptions.numPayloadValues = 3;
    // What exceptions can the pipeline throw.
    m_PipelineCompileOptions.exceptionFlags = OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_DEBUG
    | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    // 0 corresponds to enabling custom primitives and triangles, but nothing else 
    m_PipelineCompileOptions.usesPrimitiveTypeFlags = 0; 
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
    auto optixModule = CreateModule(LumenPTConsts::gs_ShaderPathBase + "draw_solid_color.ptx");

    // Descriptor of the program group, including its type, modules it uses and the name of the entry function
    OptixProgramGroupDesc rtGroupDesc = {};
    rtGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rtGroupDesc.raygen.entryFunctionName = "__raygen__draw_solid_color";
    rtGroupDesc.raygen.module = optixModule;

    auto raygen = CreateProgramGroup(rtGroupDesc, "RayGen");

    // Create a miss program group
    OptixProgramGroupDesc msGroupDesc = {};
    msGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    msGroupDesc.miss.entryFunctionName = "__miss__MissShader";
    msGroupDesc.miss.module = optixModule;

    // Create a hit program group with only a closest hit shader
    auto miss = CreateProgramGroup(msGroupDesc, "Miss");
    OptixProgramGroupDesc htGroupDesc = {};
    htGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    htGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__HitShader";
    htGroupDesc.hitgroup.moduleCH = optixModule;

    auto hit = CreateProgramGroup(htGroupDesc, "Hit");


    OptixPipelineLinkOptions pipelineLinkOptions = {};
    // How much debug information do we want from the pipeline.
    pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
    // How far is the recursion allowed to go.
    pipelineLinkOptions.maxTraceDepth = 1;

    // The program groups to include in the pipeline, can be replaced with an std::vector
    OptixProgramGroup programGroups[] = { raygen, miss, hit};

    char log[2048];
    auto logSize = sizeof(log);

    optixPipelineCreate(m_DeviceContext, &m_PipelineCompileOptions, &pipelineLinkOptions, programGroups,
        sizeof(programGroups) / sizeof(OptixProgramGroup), log, &logSize, &m_Pipeline);

    OptixStackSizes stackSizes = {};

    AccumulateStackSizes(raygen, stackSizes);
    AccumulateStackSizes(miss, stackSizes);
    AccumulateStackSizes(hit, stackSizes);

    auto finalSizes = ComputeStackSizes(stackSizes, 1, 0, 0);

    optixPipelineSetStackSize(m_Pipeline, finalSizes.DirectSizeTrace,
        finalSizes.DirectSizeState, 2048, 3);

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

std::unique_ptr<AccelerationStructure> OptiXRenderer::BuildGeometryAccelerationStructure(
    const OptixAccelBuildOptions& a_BuildOptions,
    const OptixBuildInput& a_BuildInput)
{
    
    // Let Optix compute how much memory the output buffer and the temporary buffer need to have, then create these buffers
    OptixAccelBufferSizes sizes;
    optixAccelComputeMemoryUsage(m_DeviceContext, &a_BuildOptions, &a_BuildInput, 1, &sizes);
    MemoryBuffer tempBuffer(sizes.tempSizeInBytes);
    std::unique_ptr<MemoryBuffer> resultBuffer = std::make_unique<MemoryBuffer>(sizes.outputSizeInBytes);

    OptixTraversableHandle handle;
    // It is possible to have functionality similar to DX12 Command lists with cuda streams, though it is not required.
    // Just worth mentioning if we want that functionality.
    optixAccelBuild(m_DeviceContext, 0, &a_BuildOptions, &a_BuildInput, 1, *tempBuffer, sizes.tempSizeInBytes,
        **resultBuffer, sizes.outputSizeInBytes, &handle, nullptr, 0);

    // Needs to return the resulting memory buffer and the traversable handle
    return std::make_unique<AccelerationStructure>(handle, std::move(resultBuffer));
}

std::unique_ptr<AccelerationStructure> OptiXRenderer::BuildInstanceAccelerationStructure(
    std::vector<OptixInstance> a_Instances)
{
    // The fucky part here is figuring out how the build the a_Instances vector outside this function
    // since it will contain SBT indices, transforms and whatnot. Those will probably come from the static mesh instances we have in the scene

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
    optixAccelComputeMemoryUsage(m_DeviceContext, &buildOptions, &buildInput, 1, &sizes);

    auto tempBuffer = MemoryBuffer(sizes.tempSizeInBytes);
    auto resultBuffer = std::make_unique<MemoryBuffer>(sizes.outputSizeInBytes);

    OptixTraversableHandle handle = 0;
    optixAccelBuild(m_DeviceContext, 0, &buildOptions, &buildInput, 1, *tempBuffer, tempBuffer.GetSize(),
        **resultBuffer, resultBuffer->GetSize(), &handle, nullptr, 0);

    auto res = std::make_unique<AccelerationStructure>(handle, std::move(resultBuffer));

    // Needs to return the resulting memory buffer and the traversable handle
    return res;
}

void OptiXRenderer::CreateOutputBuffer()
{
    m_OutputBuffer = std::make_unique<OutputBuffer>(gs_ImageWidth, gs_ImageHeight);
}

void OptiXRenderer::DebugCallback(unsigned int a_Level, const char* a_Tag, const char* a_Message, void*)
{
    if (a_Level == 2)
    {
        __debugbreak();
    }
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

void OptiXRenderer::CreateShaderBindingTable()
{
    // OptixShaderBindingTable is just a struct which describes where the different parts of the table are located in GPU memory
    // This means that the entire table could easily be split into multiple buffers if we so desire

    m_RayGenRecord = m_ShaderBindingTableGenerator->SetRayGen<RaygenData>();

    auto& rayGenRecord = m_RayGenRecord.GetRecord();

    rayGenRecord.m_Header = GetProgramGroupHeader("RayGen");
    rayGenRecord.m_Data.m_Color = { 0.4f, 0.5f, 0.2f };
    

    m_MissRecord = m_ShaderBindingTableGenerator->AddMiss<MissData>();
    
    auto& missRecord = m_MissRecord.GetRecord();
    missRecord.m_Header = GetProgramGroupHeader("Miss");
    missRecord.m_Data.m_Num = 2;
    missRecord.m_Data.m_Color = { 0.0f, 0.0f, 1.0f };


    m_HitRecord = m_ShaderBindingTableGenerator->AddHitGroup<HitData>();

    auto& hitRecord = m_HitRecord.GetRecord();

    hitRecord.m_Header = GetProgramGroupHeader("Hit");
    hitRecord.m_Data.m_TextureObject = **m_Texture;
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
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    m_TempBuffers.clear();


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
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    LaunchParameters params = {};
    m_Camera.SetPosition(glm::vec3(0.0f, 0.0f, -50.0f));
    auto trv = BuildGeometryAccelerationStructure(verti);


    auto& gtrv = static_cast<PTPrimitive*>(m_Scene->m_MeshInstances[0]->GetMesh().get()->m_Primitives[0].get())->m_GeometryAccelerationStructure;
    std::vector<OptixInstance> instances;
    {
        auto& inst = instances.emplace_back();
        inst.visibilityMask = 255;
        inst.sbtOffset = 0;
        inst.traversableHandle = gtrv->m_TraversableHandle;
        inst.flags = OPTIX_INSTANCE_FLAG_NONE;
        inst.instanceId = 1;

        glm::mat4 identity = glm::mat4(1.0f);  
        memcpy(&inst.transform, &identity, sizeof(inst.transform));
    }

    auto itrv = BuildInstanceAccelerationStructure(instances);

    glm::mat4 identity = glm::transpose(m_TestTransform.GetTransformationMatrix());

    instances[0].traversableHandle = itrv->m_TraversableHandle;
    memcpy(&instances[0].transform, &identity, sizeof(instances[0].transform));
    auto iitrv = BuildInstanceAccelerationStructure(instances);


    params.m_Image = m_OutputBuffer->GetDevicePointer();
    params.m_Handle = static_cast<PTScene&>(*m_Scene).GetSceneAccelerationStructure();
    params.m_ImageWidth = gs_ImageWidth;
    params.m_ImageHeight = gs_ImageHeight;
    params.m_VertexBuffer = vertexBuffer.GetDevicePtr<Vertex>();
    // Fill out struct here with whatev

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    m_Camera.SetAspectRatio(static_cast<float>(gs_ImageWidth) / static_cast<float>(params.m_ImageHeight));
    glm::vec3 eye, U, V, W;
    m_Camera.GetVectorData(eye, U, V, W);

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    params.eye = make_float3(eye.x, eye.y, eye.z);
    params.U   = make_float3(U.x, U.y, U.z);
    params.V   = make_float3(V.x, V.y, V.z);
    params.W   = make_float3(W.x, W.y, W.z);


    err = cudaGetLastError();

    MemoryBuffer devBuffer(sizeof(params));
    devBuffer.Write(params);

    err = cudaGetLastError();
    auto c = m_MissRecord.GetRecord().m_Data.m_Color;

    static float f = 0.0f;

    f += 1.0f / 60.0f;

    m_MissRecord.GetRecord().m_Data.m_Color = { 0.0f, sinf(f), 0.0f };

    OptixShaderBindingTable sbt = m_ShaderBindingTableGenerator->GetTableDesc();
    err = cudaGetLastError();

    optixLaunch(m_Pipeline, 0, *devBuffer, devBuffer.GetSize(), &sbt, params.m_ImageWidth, params.m_ImageHeight, 3);

    err = cudaGetLastError();

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

std::unique_ptr<Lumen::ILumenPrimitive> OptiXRenderer::CreatePrimitive(PrimitiveData& a_PrimitiveData)
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

    cudaDeviceSynchronize();
    auto err = cudaGetLastError();

    return std::make_unique<MemoryBuffer>(vertices);
}

std::shared_ptr<Lumen::ILumenMesh> OptiXRenderer::CreateMesh(std::vector<std::unique_ptr<Lumen::ILumenPrimitive>>& a_Primitives)
{
    auto mesh = std::make_shared<PTMesh>(a_Primitives, m_ServiceLocator);
    return mesh;
}

std::shared_ptr<Lumen::ILumenTexture> OptiXRenderer::CreateTexture(void* a_RawData, uint32_t a_Width, uint32_t a_Height)
{
    static cudaChannelFormatDesc formatDesc = cudaCreateChannelDesc<uchar4>();
    return std::make_shared<Texture>(a_RawData, formatDesc, a_Width, a_Height);
}

std::shared_ptr<Lumen::ILumenMaterial> OptiXRenderer::CreateMaterial(const MaterialData& a_MaterialData)
{
    auto mat = std::make_shared<Material>();
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
    std::shared_ptr<Lumen::ILumenVolume> volume = std::make_shared<PTVolume>(a_FilePath, m_ServiceLocator);

    return volume;
}