#pragma once

#include "SceneDataTableEntry.h"

#include "MemoryBuffer.h"
#include <unordered_map>

#include "Cuda/cuda_runtime.h"
#include "Optix/optix_types.h"

class SceneDataTableAccessor;

class SceneDataTable
{
    friend class SceneDataTableEntryBase;

public:
    SceneDataTable()
        : m_NextEntryKey(0)
        , m_EntryStride(0)
    { }

    template<typename T>
    SceneDataTableEntry<T> AddEntry()
    {
        SceneDataTableEntry<T> handle;
        handle.m_EntryListRef = &m_Entries;
        handle.m_Key = m_NextEntryKey;
        m_Entries.emplace(m_NextEntryKey++, &handle);

        return handle;
    };

    void UpdateTable();

    SceneDataTableAccessor* GetDevicePointer();

private:

    bool UpdateStride();

    void PartialUpdate();

    void FullRebuild();

    // Keys are 64-bit so we can skip on tracking what keys are already in use and which not
    std::unordered_map<uint64_t, SceneDataTableEntryBase*> m_Entries;
    uint64_t m_NextEntryKey;

    MemoryBuffer m_GpuBuffer;
    int64_t m_EntryStride;

    MemoryBuffer m_AccessorBuffer;
};

