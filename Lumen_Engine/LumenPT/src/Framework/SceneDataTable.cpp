#include "SceneDataTable.h"

#include "../Shaders/CppCommon/SceneDataTableAccessor.h"

#include <algorithm>
#include <cassert>

#include "CudaUtilities.h"

void SceneDataTable::UpdateTable()
{
    // Check if the stride has grown, as that would require a full rebuild of the table.
    bool rebuild = UpdateStride();

    // If the number of entries in the table is too much for the allocated memory, a full rebuild is necessary in the newly allocated memory.
    // Also, fully rebuild if the stride has grown because of new entry types
    if (m_Entries.size() * m_EntryStride > m_GpuBuffer.GetSize() || rebuild)
    {
        FullRebuild();
    }
    else
    {
        // Otherwise, the cheaper, partial rebuild can be performed which only touches writes into entries which have been modified
        // TODO: Partial update is broken, needs to be reimplemented correctly
        FullRebuild();
    }
}

SceneDataTableAccessor* SceneDataTable::GetDevicePointer()
{
    // Update the data table in order to return the most recent and most valid version of it
    UpdateTable();

    // Create the accessor and upload it to its memory buffer
    SceneDataTableAccessor acc(m_EntryStride, m_GpuBuffer.GetDevicePtr());
    m_AccessorBuffer.Resize(sizeof(SceneDataTableAccessor));
    m_AccessorBuffer.Write(acc);
    // Return a pointer to the accessor buffer
    return m_AccessorBuffer.GetDevicePtr<SceneDataTableAccessor>();
}

bool SceneDataTable::UpdateStride()
{
    // Find the size of the biggest entry in the table
    uint32_t max = 0;
    for (auto& entry : m_Entries)
    {
        max = std::max(entry.second->m_Size, max);
    }

    // Check if aligning the biggest size to the required by CUDA results in the stride growing
    auto oldStride = m_EntryStride;
    m_EntryStride = (max / 16 + 1) * 16;
    return m_EntryStride > oldStride;
}

void SceneDataTable::PartialUpdate()
{
    // TODO: This is low-key scuffed, needs rewriting 
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
    // Resize the buffer if it wouldn't be big enough to fit the entire data table
    auto bufferSize = m_EntryStride * m_Entries.size();
    if (m_GpuBuffer.GetSize() < bufferSize)
    {
        m_GpuBuffer.Resize(std::max(m_GpuBuffer.GetSize() * 2, bufferSize));
    }
    CHECKLASTCUDAERROR;
    // Copy all entries into the buffer, assigning them new indices in the process
    uint32_t indexCounter = 0;
    for (auto& entryPair : m_Entries)
    {
        auto& entry = entryPair.second;
        entry->m_Dirty = false;

        auto offset = indexCounter * m_EntryStride;

        cudaMemcpy(reinterpret_cast<void*>(*m_GpuBuffer + offset), entry->m_RawData, entry->m_Size, cudaMemcpyHostToDevice);
        CHECKLASTCUDAERROR;
        entry->m_TableIndex = indexCounter++;
    }
    CHECKLASTCUDAERROR;
};