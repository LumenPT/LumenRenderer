#pragma once
#include "OptixWrapper.h"
#include "MemoryBuffer.h"

template <typename VertexType, typename IndexType>
std::unique_ptr<AccelerationStructure> WaveFront::OptixWrapper::BuildGeometryAccelerationStructure(
    std::vector<VertexType> a_Vertices,
    size_t a_VertexOffset, std::vector<IndexType> a_Indices, size_t a_IndexOffset) const
{
    // Double check if the IndexType is uint32_t or uint16_t as those are the only supported index formats
    static_assert(std::is_same<IndexType, uint32_t>::value, "The index type needs to be either a 16- or 32-bit unsigned int");

    // Upload the vertex data to the device
    MemoryBuffer vBuffer(a_Vertices.size() * sizeof(VertexType) - a_VertexOffset);
    vBuffer.Write(a_Vertices.data() + a_VertexOffset, a_Vertices.size() * sizeof(VertexType), 0);

    bool hasIndexBuffer = !a_Indices.empty();
    MemoryBuffer iBuffer(a_Indices.size() * sizeof(IndexType) - a_IndexOffset);
    if (hasIndexBuffer) // Upload the index data to the device
        iBuffer.Write(a_Indices.data() + a_IndexOffset, a_Indices.size() * sizeof(IndexType), 0);

    unsigned int flags = OPTIX_GEOMETRY_FLAG_NONE;

    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    buildInput.triangleArray.indexBuffer = hasIndexBuffer ? *iBuffer : 0;
    buildInput.triangleArray.indexStrideInBytes = static_cast<uint32_t>(a_Indices.size()); // If the buffer is empty, this is 0 and everything is fine in the universe
    if (hasIndexBuffer) // By default, the index format is set to none, so no need for an else statement
        buildInput.triangleArray.indexFormat = sizeof(IndexType) == 2 ? OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3 : OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    buildInput.triangleArray.numVertices = static_cast<uint32_t>(a_Vertices.size());
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
    buildInput.triangleArray.numSbtRecords = 1;
    buildInput.triangleArray.flags = &flags;

    OptixAccelBuildOptions buildOptions = {};
    buildOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE; // Static mesh which is build once, so we build it with FAST_TRACE
    buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD; // Obviously building
    buildOptions.motionOptions = {}; // No motion

    return BuildGeometryAccelerationStructure(buildOptions, buildInput);

}