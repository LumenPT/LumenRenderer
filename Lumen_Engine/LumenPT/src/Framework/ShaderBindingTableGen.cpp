#include "ShaderBindingTableGen.h"

#include <cassert>

void ShaderBindingTableGenerator::UpdateTable()
{
    // The shader binding table is invalid without a ray gen record, so assert that one is specified
    assert(m_RayGenRecord);
    // If the ray gen record has changed, we update the ray gen buffer's contents
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

    // Record the strides for the miss and hit group records before we update them.
    size_t hitStride, missStride;
    hitStride = m_Table.hitgroupRecordStrideInBytes;
    missStride = m_Table.missRecordStrideInBytes;

    // Update the miss and hit group strides
    UpdateStrides();

    // Check if the new strides for the miss and hit group records have grown bigger.
    // If they have grown smaller, we do not bother to do a full rebuild.
    bool hitRebuild, missRebuild;
    hitRebuild = hitStride < m_Table.hitgroupRecordStrideInBytes;
    missRebuild = missStride < m_Table.missRecordStrideInBytes;

    // If the miss records have become more than the miss record buffer can fit, or the stride has increased, we build the buffer from scratch.
    if (m_MissRecords.size() * m_Table.missRecordStrideInBytes > m_MissBuffer.GetSize() || missRebuild)
    {
        FullRebuild(m_MissBuffer, m_MissRecords, m_Table.missRecordStrideInBytes);
    }
    else
    {
        // Otherwise we only update the ones that have changed since the last update.
        PartialUpdate(m_MissBuffer, m_MissRecords, m_Table.missRecordStrideInBytes);
    }

    // Update the miss record info in the shader binding table descriptor
    m_Table.missRecordBase = *m_MissBuffer;
    m_Table.missRecordCount = m_MissRecords.size();


    // If the hit group records have become more than the miss record buffer can fit, or the stride has increased, we build the buffer from scratch.
    if (m_HitRecords.size() * m_Table.hitgroupRecordStrideInBytes> m_HitGroupBuffer.GetSize() || hitRebuild)
    {
        FullRebuild(m_HitGroupBuffer, m_HitRecords, m_Table.hitgroupRecordStrideInBytes);
    }
    else
    {
        // Otherwise we only update the ones that have changed since the last update.
        PartialUpdate(m_HitGroupBuffer, m_HitRecords, m_Table.hitgroupRecordStrideInBytes);
    }

    // Update the hit group record info in the shader binding table descriptor
    m_Table.hitgroupRecordBase = *m_HitGroupBuffer;
    m_Table.hitgroupRecordCount = m_HitRecords.size();

}

void ShaderBindingTableGenerator::UpdateStrides()
{
    // Find the biggest hit group record size
    uint32_t max = 0;
    for (auto& hitRecord : m_HitRecords)
    {
        max = std::max(hitRecord.second->m_Size, max);
    }

    // Ensure that the stride is following the optix shader binding table alignment rules
    m_Table.hitgroupRecordStrideInBytes = (max / OPTIX_SBT_RECORD_ALIGNMENT + 1) * OPTIX_SBT_RECORD_ALIGNMENT;

    // Find the biggest miss record size
    max = 0;
    for (auto& missRecord : m_MissRecords)
    {
        max = std::max(missRecord.second->m_Size, max);
    }

    // Ensure that the stride is following the optix shader binding table alignment rules
    m_Table.missRecordStrideInBytes = (max / OPTIX_SBT_RECORD_ALIGNMENT + 1) * OPTIX_SBT_RECORD_ALIGNMENT;
}

void ShaderBindingTableGenerator::PartialUpdate(MemoryBuffer& a_TargetBuffer,
    std::unordered_map<uint64_t, RecordHandleBase*>& a_Records, size_t a_RecordStride)
{
    // TODO: Scuffed, needs to be rewritten
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
    // Check if the provided buffer is too small to fit all the data from the provided records
    // Resize it if it is too small
    auto bufferSize = a_RecordStride * a_Records.size();
    if (a_TargetBuffer.GetSize() < bufferSize)
    {
        a_TargetBuffer.Resize(bufferSize);
    }

    // Go through all the records and copy them into the buffer and updating their table indices meanwhile
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

