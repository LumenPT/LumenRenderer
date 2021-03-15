#include "SceneDataTableEntry.h"

#include <algorithm>


SceneDataTableEntryBase::SceneDataTableEntryBase()
    : m_Dirty(true)
    , m_TableIndex(0)
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
    if (m_EntryListRef)
    {
        (*m_EntryListRef)[m_Key] = this;
    }
};
