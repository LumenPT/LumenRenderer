#pragma once

#include <memory>

#include "Optix/optix_types.h"

class MemoryBuffer;

// Wrapper class around an OptixTraversableHandle and it's GPU memory
// Simplifies acceleration structure storage and ensures the memory is not freed prematurely
class AccelerationStructure
{
public:
    AccelerationStructure(OptixTraversableHandle a_TraversableHandle = 0, std::unique_ptr<MemoryBuffer> a_StructureBuffer = nullptr)
        : m_TraversableHandle(a_TraversableHandle)
        , m_StructureBuffer(std::move(a_StructureBuffer))
    {};

    const OptixTraversableHandle m_TraversableHandle;
    std::unique_ptr<MemoryBuffer> m_StructureBuffer;

private:

};
