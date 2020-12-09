#pragma once

#include "ShaderBindingTableRecord.h"
#include "MemoryBuffer.h"

#include "Cuda/cuda_runtime.h"
#include "Optix/optix_types.h"

#include <algorithm>
#include <unordered_map>



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

    template<typename T>
    RecordHandle<T> SetRayGen()
    {
        RecordHandle<T> handle;
        handle.m_RecordListRef = nullptr;
        handle.m_Key = 0;
        handle.m_RecordHandleRef = &m_RayGenRecord;
        return handle;
    };

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

    void UpdateTable();

    OptixShaderBindingTable GetTableDesc()
    {
        UpdateTable();

        return m_Table;
    };

private:

    void UpdateStrides();

    void PartialUpdate(MemoryBuffer& a_TargetBuffer, std::unordered_map<uint64_t, RecordHandleBase*>& a_Records, size_t a_RecordStride);

    void FullRebuild(MemoryBuffer& a_TargetBuffer, std::unordered_map<uint64_t, RecordHandleBase*>& a_Records, size_t a_RecordStride);

    RecordHandleBase* m_RayGenRecord;

    // Keys are 64-bit so we can skip on tracking what keys are already in use and which not
    std::unordered_map<uint64_t, RecordHandleBase*> m_MissRecords;
    uint64_t m_NextMissKey;
    std::unordered_map<uint64_t, RecordHandleBase*> m_HitRecords;
    uint64_t m_NextHitKey;

    MemoryBuffer m_RayGenBuffer;
    MemoryBuffer m_MissBuffer;
    MemoryBuffer m_HitGroupBuffer;

    OptixShaderBindingTable m_Table;
};
