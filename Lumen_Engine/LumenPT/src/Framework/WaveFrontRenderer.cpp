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
#include <filesystem>
#include "Cuda/cuda.h"
#include "Cuda/cuda_runtime.h"
#include "Optix/optix_stubs.h"
#include <glm/gtx/compatibility.hpp>

const unsigned BYTES_PER_PIXEL = 3;

#pragma pack(push, 1)

struct BitmapFileHeader
{
    char m_Signature[2];
    unsigned int m_ImageFileSizeBytes;
    const unsigned int m_Reserved;
    unsigned int m_PixelArrayOffset;
};

struct BitmapInfoHeader
{
    const unsigned int m_HeaderSize = sizeof(BitmapInfoHeader);
    unsigned int m_ImageWidth;
    unsigned int m_ImageHeight;
    uint16_t m_numColorPlanes;
    uint16_t m_bitsPerPixel;
    unsigned int m_Compression;
    unsigned int m_ImageSize;
    unsigned int m_HorizontalResolution;
    unsigned int m_VerticalResolution;
    unsigned int m_numColorsInTable;
    unsigned int m_numImportantColors;
};

#pragma pack(pop)

BitmapFileHeader createBitmapFileHeader(unsigned a_Height, unsigned a_Stride)
{
    
    const unsigned int fileSize = static_cast<unsigned>(sizeof(BitmapFileHeader)) + static_cast<unsigned>(sizeof(BitmapInfoHeader)) + (a_Stride * a_Height);

    BitmapFileHeader fileHeader{};
    fileHeader.m_Signature[0] = 'B';
    fileHeader.m_Signature[1] = 'M';
    fileHeader.m_ImageFileSizeBytes = fileSize;
    fileHeader.m_PixelArrayOffset = sizeof(BitmapFileHeader) + sizeof(BitmapInfoHeader);

    return fileHeader;
}

BitmapInfoHeader createBitmapInfoHeader(unsigned a_Height, unsigned a_Width)
{

    BitmapInfoHeader infoHeader{};

    infoHeader.m_ImageWidth = a_Width;
    infoHeader.m_ImageHeight = a_Height;
    infoHeader.m_numColorPlanes = 1;
    infoHeader.m_bitsPerPixel = BYTES_PER_PIXEL * 8;

    return infoHeader;
}

void CreateDirectoryIfNotExists(const std::filesystem::path& a_SaveFileDirectory)
{

    std::error_code error;
    if (!exists(a_SaveFileDirectory))
    {

        create_directories(a_SaveFileDirectory, error);
        if (error)
        {
            printf("Error: Could not create directory with given path (%s)\n\terror message: %s \n",
                a_SaveFileDirectory.string().c_str(),
                error.message().c_str());
        }
        error.clear();

        return;

    }

    if (!is_directory(a_SaveFileDirectory, error))
    {
        printf("Error: Given path (%s) is not a directory\n\terror message: %s \n", a_SaveFileDirectory.string().c_str(), error.message().c_str());
        return;
    }

}

bool SaveToBMP(
    const unsigned char* a_ImagePtr,
    const unsigned a_Width, 
    const unsigned a_Height,
    const std::filesystem::path& a_SaveFilePath)
{

    const unsigned int widthInBytes = a_Width * BYTES_PER_PIXEL;
    const unsigned int paddingSize = (4 - widthInBytes % 4) % 4;
    const unsigned int stride = widthInBytes + paddingSize;
    const unsigned char padding[3] = { 0 };

    BitmapFileHeader fileHeader = createBitmapFileHeader(a_Height, stride);
    BitmapInfoHeader infoHeader = createBitmapInfoHeader(a_Height, a_Width);

    std::fstream fileStream{};

    fileStream.open(a_SaveFilePath, std::ios::out | std::ios::binary);

    fileStream.write(reinterpret_cast<const char*>(&fileHeader), sizeof(fileHeader));
    fileStream.write(reinterpret_cast<const char*>(&infoHeader), sizeof(infoHeader));

    if (!fileStream.is_open())
    {
        printf("Error: could not open/create file to write image - path: %s \n", a_SaveFilePath.string().c_str());
        return false;
    }

    for(unsigned row = 0; row < a_Height; ++row)
    {
        const unsigned char* pixelDataPtr = &a_ImagePtr[((a_Height -1) - row) * widthInBytes];
        fileStream.write(reinterpret_cast<const char*>(pixelDataPtr), widthInBytes);
        fileStream.write(reinterpret_cast<const char*>(&padding), paddingSize);
    }

    fileStream.close();
    return true;

}


void CreateTestBMP()
{

    const int height = 100;
    const int width = 100;
    unsigned char image[height][width][BYTES_PER_PIXEL];

    int i, j;
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            image[i][j][2] = static_cast<unsigned char>(static_cast<float>(i) / static_cast<float>(height) * 255.f);
            //image[i][j][2] = j < (width / 2) ? static_cast<unsigned char>(0) : static_cast<unsigned char>(255);
            //image[i][j][2] = static_cast<unsigned char>(0);
            image[i][j][1] = static_cast<unsigned char>(static_cast<float>(j) / static_cast<float>(width) * 255.f);
            //image[i][j][1] = i < (height / 2) ? static_cast<unsigned char>(0) : static_cast<unsigned char>(255);
            //image[i][j][1] = static_cast<unsigned char>(0);
            image[i][j][0] = static_cast<unsigned char>(static_cast<float>(i + j) / static_cast<float>(height + width) * 255.f);
            //image[i][j][0] = static_cast<unsigned char>(0);
        }
    }

    SaveToBMP(reinterpret_cast<unsigned char*>(image), width, height, "./bitmapTest.bmp");

}

void SavePixelBufferToBMP(
    void* a_GpuMemPtr,
    const unsigned int a_Width,
    const unsigned int a_Height,
    const unsigned int a_NumChannels,
    const std::filesystem::path& a_SaveFileDirectory,
    const std::string& a_SaveFileName)
{

    CreateDirectoryIfNotExists(a_SaveFileDirectory);

    const unsigned int totalPixels = a_Width * a_Height * a_NumChannels;
    const unsigned int pixelsPerChannel = a_Width * a_Height;
    const unsigned int byteSize = sizeof(WaveFront::PixelBuffer) + totalPixels * sizeof(float3);

    void* cpuMemPtr = malloc(byteSize);
    cudaMemcpy(cpuMemPtr, a_GpuMemPtr, byteSize, cudaMemcpyDeviceToHost);

    const PixelBuffer* pixelBuffer = reinterpret_cast<PixelBuffer*>(cpuMemPtr);
    const float3* pixelDataBuffer = pixelBuffer->m_Pixels;



    unsigned char* imageBuffer = reinterpret_cast<unsigned char*>(malloc(pixelsPerChannel * 3));

    for(unsigned int channelIndex = 0; channelIndex < a_NumChannels; ++ channelIndex)
    {

        //Get File path for current channel image.
        std::filesystem::path finalPath = a_SaveFileDirectory;
        finalPath /= a_SaveFileName + std::to_string(channelIndex);

        if (finalPath.has_extension()){ finalPath.replace_extension(".bmp"); }
        else { finalPath += ".bmp"; }

        if (!(finalPath.has_filename() && finalPath.has_extension()))
        {
            printf("Error: could not construct valid file path: %s \n", finalPath.string().c_str());
            return;
        }

        //Convert GPU memory layout to Image layout expected by SaveToBMP function.
        
        for (unsigned int row = 0; row < a_Height; ++row)
        {

            for (unsigned int column = 0; column < a_Width; ++column)
            {

                const unsigned pixelIndex = row * a_Width + column;
                const unsigned pixelArrIndex = pixelIndex * a_NumChannels + channelIndex; //Same pattern as is used in PixelBuffer data struct.
                const float3& currentPixel = pixelDataBuffer[pixelArrIndex];

                constexpr float maxColorChannelVal = static_cast<float>(0xFF);

                const unsigned int rowIndexOffset = row * a_Width * 3;
                const unsigned int columnIndexOffset = column * 3;
                imageBuffer[rowIndexOffset + columnIndexOffset + 0] = static_cast<unsigned char>(std::round(currentPixel.z * maxColorChannelVal));
                imageBuffer[rowIndexOffset + columnIndexOffset + 1] = static_cast<unsigned char>(std::round(currentPixel.y * maxColorChannelVal));
                imageBuffer[rowIndexOffset + columnIndexOffset + 2] = static_cast<unsigned char>(std::round(currentPixel.x * maxColorChannelVal));

            }
        }

        if(SaveToBMP(imageBuffer, a_Width, a_Height, finalPath))
        {
            printf("Info: saved Pixel buffer to %s \n", finalPath.string().c_str());
        }

    }

    if (cpuMemPtr != nullptr) { free(cpuMemPtr); }
    if (imageBuffer != nullptr) { free(imageBuffer); }

}

void SaveRayBatchToBMP(
    void* a_GpuMemPtr,
    const unsigned int a_Width,
    const unsigned int a_Height,
    const unsigned int a_SamplesPerPixel,
    const std::filesystem::path& a_SaveFileDirectory,
    const std::string& a_SaveFileName)
{

    CreateDirectoryIfNotExists(a_SaveFileDirectory);

    const unsigned int totalRays = a_Width * a_Height * a_SamplesPerPixel;
    const unsigned int numRaysPerSample = a_Width * a_Height;
    const unsigned int byteSize = sizeof(RayBatch) + totalRays * sizeof(RayData);

    void* cpuMemPtr = malloc(byteSize);
    cudaMemcpy(cpuMemPtr, a_GpuMemPtr, byteSize, cudaMemcpyDeviceToHost);

    const RayBatch* rayBatch = reinterpret_cast<RayBatch*>(cpuMemPtr);
    const RayData* rayDataBuffer = rayBatch->m_Rays;


    unsigned char* imageBuffer = reinterpret_cast<unsigned char*>(malloc(numRaysPerSample * 3));

    for (unsigned int sampleIndex = 0; sampleIndex < a_SamplesPerPixel; ++sampleIndex)
    {

        //Get File path for current channel image.
        std::filesystem::path finalPath = a_SaveFileDirectory;
        finalPath /= a_SaveFileName + std::to_string(sampleIndex);

        if (finalPath.has_extension()) { finalPath.replace_extension(".bmp"); }
        else { finalPath += ".bmp"; }

        if (!(finalPath.has_filename() && finalPath.has_extension()))
        {
            printf("Error: could not construct valid file path: %s \n", finalPath.string().c_str());
            return;
        }

        for(unsigned row = 0; row < a_Height; ++row)
        {

            for(unsigned column = 0; column < a_Width; ++column)
            {

                const unsigned rayIndex = row * a_Width + column;
                const unsigned rayDataIndex = rayIndex * a_SamplesPerPixel + sampleIndex; //Same pattern as is used in PixelBuffer data struct.
                const RayData& currentPixel = rayDataBuffer[rayDataIndex];

                constexpr float maxColorChannelVal = static_cast<float>(0xFF);

                const unsigned int rowIndexOffset = row * a_Width * 3;
                const unsigned int columnIndexOffset = column * 3;
                imageBuffer[rowIndexOffset + columnIndexOffset + 0] = 
                    static_cast<unsigned char>(std::clamp((currentPixel.m_Direction.z + 1.f) / 2.f , 0.f, 1.f) * maxColorChannelVal);
                imageBuffer[rowIndexOffset + columnIndexOffset + 1] =
                    static_cast<unsigned char>(std::clamp((currentPixel.m_Direction.y + 1.f) / 2.f , 0.f, 1.f) * maxColorChannelVal);
                imageBuffer[rowIndexOffset + columnIndexOffset + 2] =
                    static_cast<unsigned char>(std::clamp((currentPixel.m_Direction.x + 1.f) / 2.f , 0.f, 1.f) * maxColorChannelVal);

            }

        }

        if(SaveToBMP(imageBuffer, a_Width, a_Height, finalPath))
        {
            printf("Info: saved Ray Batch to %s \n", finalPath.string().c_str());
        }

    }

    if (cpuMemPtr != nullptr) { free(cpuMemPtr); }
    if (imageBuffer != nullptr) { free(imageBuffer); }

}

void SaveShadowRayBatchToBMP(
    void* a_GpuMemPtr,
    const unsigned int a_Width,
    const unsigned int a_Height,
    const unsigned int a_SamplesPerPixel,
    const std::filesystem::path& a_SaveFileDirectory,
    const std::string& a_SaveFileName)
{

    CreateDirectoryIfNotExists(a_SaveFileDirectory);

    const unsigned int totalRays = a_Width * a_Height * a_SamplesPerPixel;
    const unsigned int numRaysPerSample = a_Width * a_Height;
    const unsigned int byteSize = sizeof(ShadowRayBatch) + totalRays * sizeof(ShadowRayData);

    void* cpuMemPtr = malloc(byteSize);
    cudaMemcpy(cpuMemPtr, a_GpuMemPtr, byteSize, cudaMemcpyDeviceToHost);

    const ShadowRayBatch* rayBatch = reinterpret_cast<ShadowRayBatch*>(cpuMemPtr);
    const ShadowRayData* rayDataBuffer = rayBatch->m_ShadowRays;


    unsigned char* imageBuffer = reinterpret_cast<unsigned char*>(malloc(numRaysPerSample * 3));

    for (unsigned int sampleIndex = 0; sampleIndex < a_SamplesPerPixel; ++sampleIndex)
    {

        //Get File path for current channel image.
        std::filesystem::path finalPath = a_SaveFileDirectory;
        finalPath /= a_SaveFileName + std::to_string(sampleIndex);

        if (finalPath.has_extension()) { finalPath.replace_extension(".bmp"); }
        else { finalPath += ".bmp"; }

        if (!(finalPath.has_filename() && finalPath.has_extension()))
        {
            printf("Error: could not construct valid file path: %s \n", finalPath.string().c_str());
            return;
        }

        for (unsigned row = 0; row < a_Height; ++row)
        {

            for (unsigned column = 0; column < a_Width; ++column)
            {

                const unsigned rayIndex = row * a_Width + column;
                const unsigned rayDataIndex = rayIndex * a_SamplesPerPixel + sampleIndex; //Same pattern as is used in PixelBuffer data struct.
                const ShadowRayData& currentPixel = rayDataBuffer[rayDataIndex];

                constexpr float maxColorChannelVal = static_cast<float>(0xFF);

                const unsigned int rowIndexOffset = row * a_Width * 3;
                const unsigned int columnIndexOffset = column * 3;
                imageBuffer[rowIndexOffset + columnIndexOffset + 0] =
                    static_cast<unsigned char>(std::clamp((currentPixel.m_Direction.z + 1.f) / 2.f , 0.f, 1.f) * maxColorChannelVal);
                imageBuffer[rowIndexOffset + columnIndexOffset + 1] =
                    static_cast<unsigned char>(std::clamp((currentPixel.m_Direction.y + 1.f) / 2.f , 0.f, 1.f) * maxColorChannelVal);
                imageBuffer[rowIndexOffset + columnIndexOffset + 2] =
                    static_cast<unsigned char>(std::clamp((currentPixel.m_Direction.x + 1.f) / 2.f , 0.f, 1.f) * maxColorChannelVal);

            }

        }

        if (SaveToBMP(imageBuffer, a_Width, a_Height, finalPath))
        {
            printf("Info: saved Ray Batch to %s \n", finalPath.string().c_str());
        }

    }

    if (cpuMemPtr != nullptr) { free(cpuMemPtr); }
    if (imageBuffer != nullptr) { free(imageBuffer); }

}

void SaveIntersectionBufferToBMP(
    void* a_GpuMemPtr,
    const unsigned int a_Width,
    const unsigned int a_Height,
    const unsigned int a_NumIntersectionsPerPixel,
    const std::filesystem::path& a_SaveFileDirectory,
    const std::string& a_SaveFileName)
{

    CreateDirectoryIfNotExists(a_SaveFileDirectory);

    const unsigned int totalIntersections = a_Width * a_Height * a_NumIntersectionsPerPixel;
    const unsigned int numPixels = a_Width * a_Height;
    const unsigned int byteSize = sizeof(IntersectionBuffer) + numPixels * sizeof(IntersectionData);

    void* cpuMemPtr = malloc(byteSize);
    cudaMemcpy(cpuMemPtr, a_GpuMemPtr, byteSize, cudaMemcpyDeviceToHost);

    const IntersectionBuffer* intersectionBuffer = reinterpret_cast<IntersectionBuffer*>(cpuMemPtr);
    const IntersectionData* intersectionData = intersectionBuffer->m_Intersections;

    unsigned char* imageBuffer = reinterpret_cast<unsigned char*>(malloc(numPixels * 3));

    for (unsigned int sampleIndex = 0; sampleIndex < a_NumIntersectionsPerPixel; ++sampleIndex)
    {

        //Get File path for current channel image.
        std::filesystem::path finalPath = a_SaveFileDirectory;
        finalPath /= a_SaveFileName + std::to_string(sampleIndex);

        if (finalPath.has_extension()) { finalPath.replace_extension(".bmp"); }
        else { finalPath += ".bmp"; }

        if (!(finalPath.has_filename() && finalPath.has_extension()))
        {
            printf("Error: could not construct valid file path: %s \n", finalPath.string().c_str());
            return;
        }

        for (unsigned row = 0; row < a_Height; ++row)
        {

            for (unsigned column = 0; column < a_Width; ++column)
            {

                const unsigned intersectionIndex = row * a_Width + column;
                const unsigned intersectionDataIndex = intersectionIndex * a_NumIntersectionsPerPixel + sampleIndex; //Same pattern as is used in PixelBuffer data struct.
                const IntersectionData& currentIntersection = intersectionData[intersectionIndex];

                constexpr float maxColorChannelVal = static_cast<float>(0xFF);
                const float minDistance = 0.001f;
                const float maxDistance = 1000.f;

                const unsigned int rowIndexOffset = row * a_Width * 3;
                const unsigned int columnIndexOffset = column * 3;

                if(currentIntersection.IsIntersection())
                {

                    const float normalizedDistance = std::clamp((currentIntersection.m_IntersectionT - minDistance) / (maxDistance - minDistance), 0.f, 1.f);
                    const unsigned char color = static_cast<unsigned char>(normalizedDistance * maxColorChannelVal);

                    imageBuffer[rowIndexOffset + columnIndexOffset + 0] = color;
                    imageBuffer[rowIndexOffset + columnIndexOffset + 1] = color;
                    imageBuffer[rowIndexOffset + columnIndexOffset + 2] = color;

                }
                else
                {
                    imageBuffer[rowIndexOffset + columnIndexOffset + 0] = static_cast<unsigned char>(0);
                    imageBuffer[rowIndexOffset + columnIndexOffset + 1] = static_cast<unsigned char>(0);
                    imageBuffer[rowIndexOffset + columnIndexOffset + 2] = static_cast<unsigned char>(255);
                }

            }

        }

        if (SaveToBMP(imageBuffer, a_Width, a_Height, finalPath))
        {
            printf("Info: saved Ray Batch to %s \n", finalPath.string().c_str());
        }

    }
}

const std::string cDebugOutputInitPath = "./DebugOutputs/Initialization/";
const std::string cDebugOutputRunPath = "./DebugOutputs/Run/";



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
m_RayBatches(),
m_IntersectionBuffers(),
m_ShadowRayBatch(nullptr),
m_LightBufferTemp(nullptr),
m_Texture(nullptr),
m_RenderResolution(max(a_InitializationData.m_RenderResolution, s_minResolution)),
m_OutputResolution(max(a_InitializationData.m_OutputResolution, s_minResolution)),
m_MaxDepth(max(a_InitializationData.m_MaxDepth, s_minDepth)),
m_RaysPerPixel(max(a_InitializationData.m_RaysPerPixel, s_minRaysPerPixel)),
m_ShadowRaysPerPixel(max(a_InitializationData.m_ShadowRaysPerPixel, s_minShadowRaysPerPixel)),
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
    success &= CreatePipelines(shaderPath);
    CreatePipelineBuffers();
    CreateOutputBuffer();
    CreateDataBuffers();
    SetupInitialBufferIndices();

    cudaDeviceSynchronize();

    CHECKLASTCUDAERROR;

    CreateTestBMP();

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
    const std::string resolveRaysMissFuncName = "__miss__ResolveRaysMiss";

    const std::string resolveShadowRaysParams = "resolveShadowRaysParams";
    const std::string resolveShadowRaysRayGenFuncName = "__raygen__ResolveShadowRaysRayGen";
    const std::string resolveShadowRaysHitFuncName = "__anyhit__ResolveShadowRaysAnyHit";
    const std::string resolveShadowRaysMissFuncName = "__miss__ResolveShadowRaysMiss";

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

    success &= CreatePipeline(
        shaderModule,
        compileOptions,
        PipelineType::RESOLVE_SHADOW_RAYS, 
        resolveShadowRaysRayGenFuncName, 
        resolveShadowRaysHitFuncName,
        resolveShadowRaysMissFuncName,
        m_PipelineShadowRays);

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

                msGroupDesc.miss.entryFunctionName = a_MissFuncName.c_str();
                msGroupDesc.miss.module = a_Module;

                rayGenProgram = CreateProgramGroup(rgGroupDesc, s_RaysRayGenPGName);
                hitProgram = CreateProgramGroup(htGroupDesc, s_RaysHitPGName);
                missProgram = CreateProgramGroup(msGroupDesc, s_RaysMissPGName);

                break;
            }

        case PipelineType::RESOLVE_SHADOW_RAYS:
            {
                htGroupDesc.hitgroup.entryFunctionNameAH = a_HitFuncName.c_str();
                htGroupDesc.hitgroup.moduleAH = a_Module;

                msGroupDesc.miss.entryFunctionName = a_MissFuncName.c_str();
                msGroupDesc.miss.module = a_Module;

                rayGenProgram = CreateProgramGroup(rgGroupDesc, s_ShadowRaysRayGenPGName);
                hitProgram = CreateProgramGroup(htGroupDesc, s_ShadowRaysHitPGName);
                missProgram = CreateProgramGroup(msGroupDesc, s_ShadowRaysMissPGName);

                break;
            }
        default:
            {
                rayGenProgram = nullptr;
                hitProgram = nullptr;
                missProgram = nullptr;

            }
            break;
    }

    if( rayGenProgram == nullptr ||
        hitProgram == nullptr || 
        missProgram == nullptr)
    {
        printf("Could not create program groups for pipeline type: %i: (RayGenProgram: %p , HitProgram: %p, MissProgram: %p) \n",
            a_Type,
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
    moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;

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

    if(error)
    {
        puts(log);
        abort();
    }

    return module;

}

OptixProgramGroup WaveFrontRenderer::CreateProgramGroup(OptixProgramGroupDesc a_ProgramGroupDesc, const std::string& a_Name)
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

    if(error)
    {
        puts(log);
        abort();
    }

    m_ProgramGroups.emplace(a_Name, programGroup);

    return programGroup;

}

void WaveFrontRenderer::CreateShaderBindingTables()
{

    //Do these need a data struct if there is no data needed per "shader"??
    
    m_RaysRayGenRecord = m_RaysSBTGenerator->SetRayGen<ResolveRaysRayGenData>();
    //m_RaysHitRecord = m_RaysSBTGenerator->AddHitGroup<ResolveRaysHitData>();
    m_RaysMissRecord = m_RaysSBTGenerator->AddMiss<ResolveRaysMissData>();

    auto& raysRayGenRecord = m_RaysRayGenRecord.GetRecord();
    raysRayGenRecord.m_Header = GetProgramGroupHeader(s_RaysRayGenPGName);
    raysRayGenRecord.m_Data.m_MinDistance = s_MinTraceDistance;
    raysRayGenRecord.m_Data.m_MaxDistance = s_MaxTraceDistance;

    //auto& raysHitRecord = m_RaysHitRecord.GetRecord();
    //raysHitRecord.m_Header = GetProgramGroupHeader(s_RaysHitPGName);

    auto& raysMissRecord = m_RaysMissRecord.GetRecord();
    raysMissRecord.m_Header = GetProgramGroupHeader(s_RaysMissPGName);

    m_ShadowRaysRayGenRecord = m_ShadowRaysSBTGenerator->SetRayGen<ResolveShadowRaysRayGenData>();
    m_ShadowRaysHitRecord = m_ShadowRaysSBTGenerator->AddHitGroup<ResolveShadowRaysHitData>();
    m_ShadowRaysMissRecord = m_ShadowRaysSBTGenerator->AddMiss<ResolveShadowRaysMissData>();

    auto& shadowRaysRayGenRecord = m_ShadowRaysRayGenRecord.GetRecord();
    shadowRaysRayGenRecord.m_Header = GetProgramGroupHeader(s_ShadowRaysRayGenPGName);

    auto& shadowRaysHitRecord = m_ShadowRaysHitRecord.GetRecord();
    shadowRaysHitRecord.m_Header = GetProgramGroupHeader(s_ShadowRaysHitPGName);

    auto& shadowRaysMissRecord = m_ShadowRaysMissRecord.GetRecord();
    shadowRaysMissRecord.m_Header = GetProgramGroupHeader(s_ShadowRaysMissPGName);

}

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

    void* pixelBuffMultiChannelCuPtr = reinterpret_cast<void*>(*(*m_PixelBufferMultiChannel));
    SavePixelBufferToBMP(
        pixelBuffMultiChannelCuPtr,
        m_RenderResolution.x,
        m_RenderResolution.y,
        numOutputChannels,
        cDebugOutputInitPath + "MultiPixelBuffer/",
        "MultiChannelPixelBuffer-Initialized");

    m_PixelBufferSingleChannel = std::make_unique<MemoryBuffer>(
        static_cast<size_t>(pixelBufferEmptySize) + 
        static_cast<size_t>(numPixels)*
        static_cast<size_t>(pixelDataStructSize));
    m_PixelBufferSingleChannel->Write(numPixels, 0);
    m_PixelBufferSingleChannel->Write(1, sizeof(PixelBuffer::m_NumPixels));

    void* pixelBuffSingleChannelCuPtr = reinterpret_cast<void*>(*(*m_PixelBufferSingleChannel));
    SavePixelBufferToBMP(
        pixelBuffSingleChannelCuPtr,
        m_RenderResolution.x,
        m_RenderResolution.y,
        1,
        cDebugOutputInitPath + "SinglePixelBuffer/",
        "SingleChannelPixelBuffer-Initialized");

    const PixelBuffer* pixelBufferPtr = m_PixelBufferMultiChannel->GetDevicePtr<PixelBuffer>();

    //Allocate result buffer.
    m_ResultBuffer = std::make_unique<MemoryBuffer>(sizeof(ResultBuffer));
    m_ResultBuffer->Write(&pixelBufferPtr, sizeof(ResultBuffer::m_PixelBuffer), 0);

    const unsigned rayBatchEmptySize = sizeof(RayBatch);
    const unsigned rayDataStructSize = sizeof(RayData);

    //Allocate and initialize ray batches.
    int batchIndex = 0;
    for(auto& rayBatch : m_RayBatches)
    {
        rayBatch = std::make_unique<MemoryBuffer>(
            static_cast<size_t>(rayBatchEmptySize) + 
            static_cast<size_t>(numPixels) * 
            static_cast<size_t>(m_RaysPerPixel) * 
            static_cast<size_t>(rayDataStructSize));
        //rayBatch->Write(numPixels, 0);
        //rayBatch->Write(m_RaysPerPixel, sizeof(RayBatch::m_NumPixels));

        ResetRayBatch(rayBatch->GetDevicePtr<RayBatch>(), numPixels, m_RaysPerPixel);

        void* rayBatchCuPtr = rayBatch->GetDevicePtr();
        SaveRayBatchToBMP(
            rayBatchCuPtr,
            m_RenderResolution.x,
            m_RenderResolution.y,
            m_RaysPerPixel,
            cDebugOutputInitPath + "RayBatches/",
            "RayBatch" + std::to_string(batchIndex) + "-Initialized");
        batchIndex++;

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

        void* hitBufferCuPtr = intersectionBuffer->GetDevicePtr();
        SaveIntersectionBufferToBMP(
            hitBufferCuPtr,
            m_RenderResolution.x,
            m_RenderResolution.y,
            m_RaysPerPixel,
            cDebugOutputInitPath + "HitBuffers/",
            "HitBuffer" + std::to_string(bufferIndex) + "-Initialized");
        bufferIndex++;
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



    const unsigned LightBufferEmptySize = sizeof(LightBuffer);
    const unsigned LightDataStructSize = sizeof(TriangleLight);

    m_LightBufferTemp = std::make_unique<MemoryBuffer>(
        static_cast<size_t>(LightBufferEmptySize) +
        static_cast<size_t>(3) *
        static_cast<size_t>(LightDataStructSize));


    TriangleLight lights[3] = {
        {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}},
        {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}},
        {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}} };
    LightBuffer tempLightBuffer(3,lights);


    m_LightBufferTemp->Write(
        &tempLightBuffer, 
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

    static unsigned int frameCount = 0;
    const std::string frameSaveFilePath = cDebugOutputRunPath + "Frame" + std::to_string(frameCount) + "/";

    //Clear Pixel buffer
    PixelBuffer* pixelBufferMultiChannelDevPtr = m_PixelBufferMultiChannel->GetDevicePtr<PixelBuffer>();
    const unsigned numPixels = m_RenderResolution.x * m_RenderResolution.y;
    const unsigned channelsPerPixel = static_cast<unsigned>(ResultBuffer::s_NumOutputChannels);
    ResetPixelBuffer(pixelBufferMultiChannelDevPtr, numPixels, channelsPerPixel);

    //Generate Camera rays using CUDA kernel.
    float3 eye, u, v, w;
    m_Camera.GetVectorData(eye, u, v, w);
    const WaveFront::DeviceCameraData cameraData(eye, u, v, w);

    //Get new Ray Batch to fill with Primary Rays (Either first or last ray batch, opposite of current PrimRaysPrevFrame batch)
    //Get a new array of indices to temporarily update the current array of indices.
    std::array<unsigned, s_NumRayBatchTypes> batchIndices{};
    GetRayBatchIndices(0, m_RayBatchIndices, batchIndices);

    //Get index to use to get the ray batch to use for the current ray buffer.
    const unsigned currentRayBatchIndex = batchIndices[static_cast<unsigned>(RayBatchTypeIndex::CURRENT_RAYS)];
    MemoryBuffer& currentRaysBatch = *m_RayBatches[currentRayBatchIndex];

    //Generate primary rays using the setup parameters
    const SetupLaunchParameters setupParams(m_RenderResolution, cameraData, currentRaysBatch.GetDevicePtr<RayBatch>());
    GenerateRays(setupParams);

    

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
    ResolveRaysLaunchParameters optixRaysLaunchParams{};
    optixRaysLaunchParams.m_Common.m_Traversable = dynamic_cast<PTScene&>(*m_Scene).GetSceneAccelerationStructure();

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

        MemoryBuffer& primRaysPrevFrame =   *m_RayBatches[m_RayBatchIndices[static_cast<unsigned>(RayBatchTypeIndex::PRIM_RAYS_PREV_FRAME)]];
        MemoryBuffer& currentRays =         *m_RayBatches[m_RayBatchIndices[static_cast<unsigned>(RayBatchTypeIndex::CURRENT_RAYS)]];
        MemoryBuffer& secondaryRays =       *m_RayBatches[m_RayBatchIndices[static_cast<unsigned>(RayBatchTypeIndex::SECONDARY_RAYS)]];

        MemoryBuffer& primHitsPrevFrame =   *m_IntersectionBuffers[m_HitBufferIndices[static_cast<unsigned>(HitBufferTypeIndex::PRIM_HITS_PREV_FRAME)]];
        MemoryBuffer& currentHits =         *m_IntersectionBuffers[m_HitBufferIndices[static_cast<unsigned>(HitBufferTypeIndex::CURRENT_HITS)]];

        void* primRayPrevFrameBatchCuPtr = primRaysPrevFrame.GetDevicePtr();
        void* currentRayBatchCuPtr = currentRays.GetDevicePtr();
        void* secondaryRayBatchCuPtr = secondaryRays.GetDevicePtr();


        {   //Debug stuff
            /*const std::string frameWaveSaveFilePath = frameSaveFilePath + "Wave" + std::to_string(waveIndex) + "/";*/

            /*SaveRayBatchToBMP(
                primRayPrevFrameBatchCuPtr,
                m_RenderResolution.x,
                m_RenderResolution.y,
                m_RaysPerPixel,
                frameWaveSaveFilePath + "RayBatches/",
                "PrimaryRays-Previous-Frame");*/

            /*SaveRayBatchToBMP(
                currentRayBatchCuPtr,
                m_RenderResolution.x,
                m_RenderResolution.y,
                m_RaysPerPixel,
                frameWaveSaveFilePath + "RayBatches/",
                "CurrentRays");*/

            /*SaveRayBatchToBMP(
                secondaryRayBatchCuPtr,
                m_RenderResolution.x,
                m_RenderResolution.y,
                m_RaysPerPixel,
                frameWaveSaveFilePath + "RayBatches/",
                "SecondaryRays");*/
        }

        //Resolution and current depth(, current depth = current wave index)
        resolutionAndDepth.z = waveIndex;

        optixRaysLaunchParams.m_Common.m_ResolutionAndDepth = resolutionAndDepth;
        optixRaysLaunchParams.m_Rays = currentRays.GetDevicePtr<RayBatch>();
        optixRaysLaunchParams.m_Intersections = currentHits.GetDevicePtr<IntersectionBuffer>();

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
            primRaysPrevFrame.GetDevicePtr<RayBatch>(),
            primHitsPrevFrame.GetDevicePtr<IntersectionBuffer>(),
            currentRays.GetDevicePtr<RayBatch>(),
            currentHits.GetDevicePtr<IntersectionBuffer>(),
            secondaryRays.GetDevicePtr<RayBatch>(),
            m_ShadowRayBatch->GetDevicePtr<ShadowRayBatch>(),
            m_LightBufferTemp->GetDevicePtr<LightBuffer>(),
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

    frameCount++;

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





std::unique_ptr<AccelerationStructure> WaveFrontRenderer::BuildGeometryAccelerationStructure(
    const OptixAccelBuildOptions& a_BuildOptions,
    const OptixBuildInput& a_BuildInput)
{

    // Let Optix compute how much memory the output buffer and the temporary buffer need to have, then create these buffers
    OptixAccelBufferSizes sizes{};
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