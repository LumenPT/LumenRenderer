#pragma once

#include "Optix/optix.h"

#include "cuda_runtime.h"

#include "glad/glad.h"

#include <cassert>
#include <map>
#include <string>
#include <vector>
#include <memory>

struct PipelineParameters
{
    void* m_OutputImage;
    int m_ImageWidth;
    int m_ImageHeight;
};

struct ProgramGroupHeader
{
    unsigned char m_Data[OPTIX_SBT_RECORD_HEADER_SIZE];
};

template<typename DataType>
struct SBTRecord
{
    ProgramGroupHeader m_Header;
    DataType m_Data;
};

class MemoryBuffer
{
public:

    MemoryBuffer(size_t a_Size)
        : m_Size(a_Size)
    {
        cudaMalloc(reinterpret_cast<void**>(m_CudaPtr), m_Size);
    };
    ~MemoryBuffer()
    {
        cudaFree(&m_CudaPtr);
    };

    CUdeviceptr& operator*() { return m_CudaPtr; }

    template<typename DataType>
    void Write(const DataType& a_Data, size_t a_Offset = 0)
    {
        Write(&a_Data, sizeof(DataType), a_Offset);
    }

    void Write(const void* a_Data, size_t a_Size, size_t a_Offset)
    {
        assert(a_Size + a_Offset <= m_Size);
        cudaMemcpy(reinterpret_cast<void*>(m_CudaPtr + a_Offset), a_Data, a_Size, cudaMemcpyKind::cudaMemcpyHostToDevice);
    };

    void Read(void* a_Dst, size_t a_ReadSize, size_t a_SrcOffset) const
    {
        assert(a_SrcOffset + a_ReadSize < m_Size);
        cudaMemcpy(a_Dst, reinterpret_cast<void*>(m_CudaPtr + a_SrcOffset), a_ReadSize, cudaMemcpyDeviceToHost);
    };

    void CopyFrom(MemoryBuffer a_MemoryBuffer, size_t a_Size, size_t a_DstOffset = 0, size_t a_SrcOffset = 0)
    {
        assert(a_DstOffset + a_Size < m_Size);
        assert(a_SrcOffset + a_Size < a_MemoryBuffer.GetSize());

        cudaMemcpy(reinterpret_cast<void*>(m_CudaPtr + a_DstOffset), reinterpret_cast<void*>(*a_MemoryBuffer + a_SrcOffset),
            a_Size, cudaMemcpyDeviceToDevice);
    };

    size_t GetSize() const { return m_Size; }

private:
    CUdeviceptr m_CudaPtr;
    size_t m_Size;
};

class OutputBuffer
{
public:
    OutputBuffer(uint32_t a_Width, uint32_t a_Height);
    ~OutputBuffer();

    void* GetDevicePointer();

    GLuint GetTexture();

    void Resize(uint32_t a_Width, uint32_t a_Height);

private:

    void UpdateTexture();

    uint32_t m_Width;
    uint32_t m_Height;

    GLuint m_PixelBuffer;
    GLuint m_Texture;

    cudaGraphicsResource* m_CudaGraphicsResource;

};

class OptiXRenderer
{
    public:

    struct InitializationData
    {
        
    };

    OptiXRenderer(const InitializationData& a_InitializationData);

    void InitializeContext();

    void InitializePipelineOptions();

    void CreatePipeline();

    OptixModule CreateModule(const std::string& a_PtxPath);

    OptixProgramGroup CreateProgramGroup(OptixProgramGroupDesc a_ProgramGroupDesc, const std::string& a_Name);

    void CreateShaderBindingTable();

    ProgramGroupHeader GetProgramGroupHeader(const std::string& a_GroupName) const;

    //template<typename VertexType>
    //OptixTraversableHandle BuildGeometryAccelerationStructure(std::vector<VertexType> a_Vertices, size_t a_Offset)

    template<typename VertexType, typename IndexType = uint32_t>
    OptixTraversableHandle BuildGeometryAccelerationStructure(std::vector<VertexType> a_Vertices, size_t a_VertexOffset = 0,
        std::vector<IndexType> a_Indices = std::vector<IndexType>(), size_t a_IndexOffset = 0);

    OptixTraversableHandle BuildInstanceAccelerationStructure(std::vector<OptixInstance> a_Instances);

    void CreateOutputBuffer();

    GLuint TraceFrame();

private:
    static void DebugCallback(unsigned int a_Level, const char* a_Tag, const char* a_Message, void* /*extra data provided during context initialization*/);

    static void AccumulateStackSizes(OptixProgramGroup a_ProgramGroup, OptixStackSizes& a_StackSizes);

    struct ComputedStackSizes
    {
        int DirectSizeTrace;
        int DirectSizeState;
        int ContinuationSize;
    };

    static ComputedStackSizes ComputeStackSizes(OptixStackSizes a_StackSizes, int a_TraceDepth, int a_DirectDepth, int a_ContinuationDepth);

    OptixDeviceContext              m_DeviceContext;

    OptixPipelineCompileOptions     m_PipelineCompileOptions;
    OptixPipeline                   m_Pipeline;

    std::map<std::string, OptixProgramGroup> m_ProgramGroups;

    std::unique_ptr<OutputBuffer> m_OutputBuffer;
};

template <typename VertexType, typename IndexType>
OptixTraversableHandle OptiXRenderer::BuildGeometryAccelerationStructure(std::vector<VertexType> a_Vertices,
    size_t a_VertexOffset, std::vector<IndexType> a_Indices, size_t a_IndexOffset)
{
    // Double check if the IndexType is uint32_t or uint16_t as those are the only supported index formats
    static_assert(std::is_same_v<IndexType, uint32_t> || std::is_same_v<IndexType, uint16_t>);

    // Upload the vertex data to the device
    MemoryBuffer vBuffer(a_Vertices.size() * sizeof(VertexType) - a_VertexOffset);
    vBuffer.Write(a_Vertices.data() + a_VertexOffset, a_Vertices.size() * sizeof(VertexType));

    bool hasIndexBuffer = !a_Indices.empty();
    MemoryBuffer iBuffer(a_Indices.size() * sizeof(IndexType) - a_IndexOffset);
    if (hasIndexBuffer) // Upload the index data to the device
        iBuffer.Write(a_Indices.data() + a_IndexOffset, a_Indices.size() * sizeof(IndexType));

    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    buildInput.triangleArray.indexBuffer = hasIndexBuffer ? *iBuffer : 0;
    buildInput.triangleArray.indexStrideInBytes = a_Indices.size(); // If the buffer is empty, this is 0 and everything is fine in the universe
    if (hasIndexBuffer) // By default, the index format is set to none, so no need for an else statement
        buildInput.triangleArray.indexFormat = sizeof(IndexType) == 2 ? OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3 : OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    buildInput.triangleArray.numVertices = a_Vertices.size();
    buildInput.triangleArray.vertexBuffers = &*vBuffer;
    buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3; // I doubt we will ever need a different vertex format
    // If the vertex stride is set to 0, it is assumed the vertices are tightly packed and thus the size of the vertex format is taken
    buildInput.triangleArray.vertexStrideInBytes = sizeof(VertexType) <= sizeof(float) * 3 ? 0 : sizeof(VertexType);

    // Extras which are not necessary, but are here for documentation purposes
    buildInput.triangleArray.primitiveIndexOffset = 0; // Defines an offset when accessing the primitive index offset in the hit shaders
    // If the input contains multiple primitives, each with a different Material, we can specify offsets for their SBT records here
    // This could be used as a replacement for the 3-layer acceleration structure we considered earlier
    buildInput.triangleArray.sbtIndexOffsetBuffer = 0;
    buildInput.triangleArray.sbtIndexOffsetSizeInBytes = 0;
    buildInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;

    OptixAccelBuildOptions buildOptions = {};
    buildOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE; // Static mesh which is build once, so we build it with FAST_TRACE
    buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD; // Obviously building
    buildOptions.motionOptions = {}; // No motion

    // Let Optix compute how much memory the output buffer and the temporary buffer need to have, then create these buffers
    OptixAccelBufferSizes sizes;
    optixAccelComputeMemoryUsage(m_DeviceContext, &buildOptions, &buildInput, 1, &sizes);
    MemoryBuffer tempBuffer(sizes.tempSizeInBytes);
    MemoryBuffer resultBuffer(sizes.outputSizeInBytes);

    OptixTraversableHandle handle;
    // It is possible to have functionality similar to DX12 Command lists with cuda streams, though it is not required.
    // Just worth mentioning if we want that functionality.
    optixAccelBuild(m_DeviceContext, 0, &buildOptions, &buildInput, 1, *tempBuffer, sizes.tempSizeInBytes, 
        *resultBuffer, sizes.outputSizeInBytes, &handle, nullptr, 0);

    // Needs to return the resulting memory buffer and the traversable handle
    return handle;
}

