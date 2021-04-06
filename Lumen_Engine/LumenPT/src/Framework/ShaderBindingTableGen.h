#pragma once

#include "ShaderBindingTableRecord.h"
#include "MemoryBuffer.h"

#include "Cuda/cuda_runtime.h"
#include "Optix/optix_types.h"

#include <algorithm>
#include <unordered_map>

// Class to simplify the generation of the shader binding table and its descriptor
class ShaderBindingTableGenerator
{
    friend class RecordHandleBase;

public:
    ShaderBindingTableGenerator()
        : m_RayGenRecord(nullptr)
        , m_NextMissKey(0)
        , m_NextHitKey(0)
        , m_Table({})
    {}

    // The shader binding table can have only one ray gen record.
    // Calling this function will invalidate the previous ray gen record if such exists
    template<typename T>
    RecordHandle<T> SetRayGen()
    {
        RecordHandle<T> handle;
        handle.m_RecordListRef = nullptr;
        handle.m_Key = 0;
        handle.m_RecordHandleRef = &m_RayGenRecord;
        return handle;
    };

    // Add a new miss record to the table
    template<typename T>
    RecordHandle<T> AddMiss()
    {
        RecordHandle<T> handle;
        handle.m_RecordListRef = &m_MissRecords;
        handle.m_Key = m_NextMissKey;
        handle.m_RecordHandleRef = nullptr;
        m_MissRecords.emplace(m_NextMissKey++, &handle);
        return handle;
    };

    // Adds a new hit group record to the table
    template<typename T>
    RecordHandle<T> AddHitGroup()
    {
        RecordHandle<T> handle;
        handle.m_RecordListRef = &m_HitRecords;
        handle.m_Key = m_NextHitKey;
        handle.m_RecordHandleRef = nullptr;
        m_HitRecords.emplace(m_NextHitKey++, &handle);
        return handle;
    };

    // Update the shader binding table manually. Calling GetTableDesc() automatically updates the table.
    void UpdateTable();

    // Update the shader binding table and return a descriptor of it.
    OptixShaderBindingTable GetTableDesc()
    {
        UpdateTable();

        return m_Table;
    };

private:

    // Calculate the new strides for the miss and hit group records
    void UpdateStrides();

    // Update only the entries withing a table which have changed. See m_MissBuffer and m_HitGroupBuffer.
    void PartialUpdate(MemoryBuffer& a_TargetBuffer, std::unordered_map<uint64_t, RecordHandleBase*>& a_Records, size_t a_RecordStride);

    // Update an entire table, going through all its records. See m_MissBuffer and m_HitGroupBuffer.
    void FullRebuild(MemoryBuffer& a_TargetBuffer, std::unordered_map<uint64_t, RecordHandleBase*>& a_Records, size_t a_RecordStride);

    // Reference to the ray gen record used within the table
    // The reference is to an instance of a RecordHandle whose lifetime is managed by the application
    RecordHandleBase* m_RayGenRecord;

    // Keys are 64-bit so we can skip on tracking what keys are already in use and which not
    std::unordered_map<uint64_t, RecordHandleBase*> m_MissRecords;
    std::unordered_map<uint64_t, RecordHandleBase*> m_HitRecords;
    uint64_t m_NextMissKey;
    uint64_t m_NextHitKey;

    // The raygen, miss and hitgroup records are separated in three different buffers.
    // This way if only the hit groups increase in size or number, the raygen and miss records remain unaffected.
    MemoryBuffer m_RayGenBuffer; // Buffer containing the ray gen record
    MemoryBuffer m_MissBuffer; // Buffer containing all miss records
    MemoryBuffer m_HitGroupBuffer; // Buffer containing all hit group records

    OptixShaderBindingTable m_Table; // A descriptor for the entire shader binding table, as accepted by Optix for ray tracing operations.
};
