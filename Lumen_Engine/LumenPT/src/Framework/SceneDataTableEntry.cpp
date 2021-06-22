#include "SceneDataTableEntry.h"

#include <algorithm>


SceneDataTableEntryBase::SceneDataTableEntryBase()
    : m_Dirty(true) // The entries are initialized as dirty so that they can be added into the table at the next update
    , m_TableIndex(0)
    , m_TableIndexValid(false)
    , m_Size(0)
    , m_RawData(nullptr)
    , m_EntryListRef(nullptr)
    , m_Key(0)
{};

SceneDataTableEntryBase::~SceneDataTableEntryBase()
{
    
}

void SceneDataTableEntryBase::UpdateGeneratorReference()
{
    // Update the reference in the generator if this entry is valid
    if (m_EntryListRef)
    {
        (*m_EntryListRef)[m_Key] = this;
    }
};
