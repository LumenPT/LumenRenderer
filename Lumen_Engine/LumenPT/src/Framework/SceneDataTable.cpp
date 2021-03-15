#include "SceneDataTable.h"

#include "../Shaders/CppCommon/SceneDataTableAccessor.h"

#include <algorithm>
#include <cassert>

void SceneDataTable::UpdateTable()
{
    bool rebuild = UpdateStride();

    if (m_Entries.size() * m_EntryStride > m_GpuBuffer.GetSize() || rebuild)
    {
        FullRebuild();
    }
    else
    {
        PartialUpdate();
    }
}

SceneDataTableAccessor* SceneDataTable::GetDevicePointer()
{
    UpdateTable();

    SceneDataTableAccessor acc(m_EntryStride, m_GpuBuffer.GetDevicePtr());

    m_AccessorBuffer.Resize(sizeof(SceneDataTableAccessor));

    m_AccessorBuffer.Write(acc);

    //acc.GetTableEntry(3);


    return m_AccessorBuffer.GetDevicePtr<SceneDataTableAccessor>();
}

bool SceneDataTable::UpdateStride()
{
    uint32_t max = 0;
    for (auto& entry : m_Entries)
    {
        max = std::max(entry.second->m_Size, max);
    }

    auto oldStride = m_EntryStride;

    m_EntryStride = (max / OPTIX_SBT_RECORD_ALIGNMENT + 1) * OPTIX_SBT_RECORD_ALIGNMENT;
    return m_EntryStride != oldStride;
}

void SceneDataTable::PartialUpdate()
{
    uint32_t indexCounter = 0;
    for (auto& entryPair : m_Entries)
    {
        auto& entry = entryPair.second;
        if (entry->m_Dirty)
        {
            entry->m_Dirty = false;

            auto offset = indexCounter * m_EntryStride;

            cudaMemcpy(reinterpret_cast<void*>(*m_GpuBuffer + offset), entry->m_RawData, entry->m_Size, cudaMemcpyHostToDevice);
            entry->m_TableIndex = indexCounter++;
        }
    }
}

void SceneDataTable::FullRebuild()
{
    auto bufferSize = m_EntryStride * m_Entries.size();
    if (m_GpuBuffer.GetSize() < bufferSize)
    {
        m_GpuBuffer.Resize(bufferSize);
    }

    uint32_t indexCounter = 0;
    for (auto& entryPair : m_Entries)
    {
        auto& entry = entryPair.second;
        entry->m_Dirty = false;

        auto offset = indexCounter * m_EntryStride;

        cudaMemcpy(reinterpret_cast<void*>(*m_GpuBuffer + offset), entry->m_RawData, entry->m_Size, cudaMemcpyHostToDevice);
        entry->m_TableIndex = indexCounter++;
    }
};