#include "ShaderBindingTableGen.h"

#include <cassert>

void ShaderBindingTableGenerator::UpdateTable()
{
    assert(m_RayGenRecord);
    if (m_RayGenRecord->m_Dirty)
    {
        m_RayGenRecord->m_Dirty = false;

        if (m_RayGenRecord->m_Size > m_RayGenBuffer.GetSize())
        {
            m_RayGenBuffer.Resize(m_RayGenRecord->m_Size);
            m_Table.raygenRecord = *m_RayGenBuffer;
        }
        cudaMemcpy(reinterpret_cast<void*>(*m_RayGenBuffer), m_RayGenRecord->m_RawData, m_RayGenRecord->m_Size, cudaMemcpyHostToDevice);
    }

    size_t hitStride, missStride;
    hitStride = m_Table.hitgroupRecordStrideInBytes;
    missStride = m_Table.missRecordStrideInBytes;

    UpdateStrides();

    bool hitRebuild, missRebuild;
    hitRebuild = hitStride < m_Table.hitgroupRecordStrideInBytes;
    missRebuild = missStride < m_Table.missRecordStrideInBytes;

    if (m_MissRecords.size() * m_Table.missRecordStrideInBytes > m_MissBuffer.GetSize() || missRebuild)
    {
        FullRebuild(m_MissBuffer, m_MissRecords, m_Table.missRecordStrideInBytes);
    }
    else
    {
        PartialUpdate(m_MissBuffer, m_MissRecords, m_Table.missRecordStrideInBytes);
    }

    m_Table.missRecordBase = *m_MissBuffer;
    m_Table.missRecordCount = m_MissRecords.size();


    if (m_HitRecords.size() * m_Table.hitgroupRecordStrideInBytes> m_HitGroupBuffer.GetSize() || hitRebuild)
    {
        FullRebuild(m_HitGroupBuffer, m_HitRecords, m_Table.hitgroupRecordStrideInBytes);
    }
    else
    {
        PartialUpdate(m_HitGroupBuffer, m_HitRecords, m_Table.hitgroupRecordStrideInBytes);
    }

    m_Table.hitgroupRecordBase = *m_HitGroupBuffer;
    m_Table.hitgroupRecordCount = m_HitRecords.size();

}

void ShaderBindingTableGenerator::UpdateStrides()
{
    uint32_t max = 0;
    for (auto& hitRecord : m_HitRecords)
    {
        max = std::max(hitRecord.second->m_Size, max);
    }

    m_Table.hitgroupRecordStrideInBytes = (max / OPTIX_SBT_RECORD_ALIGNMENT + 1) * OPTIX_SBT_RECORD_ALIGNMENT;

    max = 0;
    for (auto& missRecord : m_MissRecords)
    {
        max = std::max(missRecord.second->m_Size, max);
    }

    m_Table.missRecordStrideInBytes = (max / OPTIX_SBT_RECORD_ALIGNMENT + 1) * OPTIX_SBT_RECORD_ALIGNMENT;
}

void ShaderBindingTableGenerator::PartialUpdate(MemoryBuffer& a_TargetBuffer,
    std::unordered_map<uint64_t, RecordHandleBase*>& a_Records, size_t a_RecordStride)
{
    uint32_t indexCounter = 0;
    for (auto& record : a_Records)
    {
        auto& rec = record.second;
        if (rec->m_Dirty)
        {
            rec->m_Dirty = false;

            auto offset = indexCounter * a_RecordStride;

            cudaMemcpy(reinterpret_cast<void*>(*a_TargetBuffer + offset), rec->m_RawData, rec->m_Size, cudaMemcpyHostToDevice);
            rec->m_TableIndex = indexCounter++;
        }
    }
}

void ShaderBindingTableGenerator::FullRebuild(MemoryBuffer& a_TargetBuffer,
    std::unordered_map<uint64_t, RecordHandleBase*>& a_Records, size_t a_RecordStride)
{
    auto bufferSize = a_RecordStride * a_Records.size();
    if (a_TargetBuffer.GetSize() < bufferSize)
    {
        a_TargetBuffer.Resize(bufferSize);
    }

    uint32_t indexCounter = 0;
    for (auto& record : a_Records)
    {
        auto& rec = record.second;
        rec->m_Dirty = false;

        auto offset = indexCounter * a_RecordStride;

        auto err = cudaMemcpy(reinterpret_cast<void*>(*a_TargetBuffer + offset), rec->m_RawData, rec->m_Size, cudaMemcpyHostToDevice);
        /*printf("FullRebuild - TargetBuffer: %p \n", reinterpret_cast<void*>(*a_TargetBuffer + offset));*/
        rec->m_TableIndex = indexCounter++;
    }
};

