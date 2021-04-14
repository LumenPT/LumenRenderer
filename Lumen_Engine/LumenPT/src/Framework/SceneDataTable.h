#pragma once

#include "SceneDataTableEntry.h"

#include "MemoryBuffer.h"
#include <unordered_map>

#include "Cuda/cuda_runtime.h"
#include "Optix/optix_types.h"

class SceneDataTableAccessor;

// Class which manages a GPU table of all mesh and volumetric data references
// This was done to separate the mesh and volumetric specific data from the shader binding table
// The alternative would have us build more acceleration structures, which we deemed too expensive
class SceneDataTable
{
    friend class SceneDataTableEntryBase;

public:
    SceneDataTable()
        : m_NextEntryKey(0)
        , m_EntryStride(0)
    {
        m_GpuBuffer.Resize(2 * 1024 * 1024);
    }

    // Add an entry to the table containing the specified data struct.
    // The returned struct manages the lifetime of the entry within the table, so it should be kept alive
    // as long as the entry needs to exist
    template<typename T>
    SceneDataTableEntry<T> AddEntry()
    {
        // Create the entry handle
        SceneDataTableEntry<T> handle;
        // The handle needs the reference to the entry list so it can remove itself from it when destroyed
        handle.m_EntryListRef = &m_Entries;
        // It also needs the key to know which member of the list to delete
        handle.m_Key = m_NextEntryKey;
        m_Entries.emplace(m_NextEntryKey++, &handle);

        return handle;
    };

    // Manually update the table. Calling GetDevicePointer will automatically update the table.
    void UpdateTable();

    // Returns a SceneDataTableAccessor* to GPU memory.
    // This can be used to easily access the table's contents from the GPU using only the instance IDs received from Optix
    SceneDataTableAccessor* GetDevicePointer();

private:

    // Ensure that the stride has not grown, and if it has, return true.
    bool UpdateStride();

    // Update only the parts of the table that have changed
    void PartialUpdate();

    // Completely rebuild the entire table
    void FullRebuild();

    // Keys are 64-bit so we can skip on tracking what keys are already in use and which not
    std::unordered_map<uint64_t, SceneDataTableEntryBase*> m_Entries;
    uint64_t m_NextEntryKey;

    // Memory buffer containing the table itself
    MemoryBuffer m_GpuBuffer;
    // Stride between the separate entries
    int64_t m_EntryStride;

    // Memory buffer containing the scene data table accessor
    MemoryBuffer m_AccessorBuffer;
};

