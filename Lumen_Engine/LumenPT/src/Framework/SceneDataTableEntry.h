#pragma once

#include <unordered_map>

class SceneDataTableEntryBase
{
    friend class SceneDataTable;
public:

    SceneDataTableEntryBase();

    virtual ~SceneDataTableEntryBase();

    SceneDataTableEntryBase(SceneDataTableEntryBase&) = delete;
    SceneDataTableEntryBase& operator=(SceneDataTableEntryBase&) = delete;

    uint32_t m_TableIndex;

protected:
    void UpdateGeneratorReference();

    void* m_RawData; // Pointer to the record in the child object
    uint32_t m_Size; // Size of the record owned by the child object

    std::unordered_map<uint64_t, SceneDataTableEntryBase*>* m_EntryListRef;

    uint64_t m_Key;
    bool m_Dirty;
    uint32_t count;
};

template<typename T>
class SceneDataTableEntry : public SceneDataTableEntryBase
{
public:
    SceneDataTableEntry()
        : SceneDataTableEntryBase()
    {
        m_Size = sizeof(T);
        m_RawData = &m_Data;
    }

    ~SceneDataTableEntry();

    SceneDataTableEntry(SceneDataTableEntry<T>&& a_Other);

    SceneDataTableEntry& operator=(SceneDataTableEntry<T>&& a_Other);

    T& GetData()
    {
        m_Dirty = true;
        return m_Data;
    };

    const T& GetConstData() const
    {
        return m_Data;
    };
private:
    T m_Data;
};

template <typename T>
SceneDataTableEntry<T>::~SceneDataTableEntry()
{
    if (m_EntryListRef)
        m_EntryListRef->erase(m_Key);
}

template <typename T>
SceneDataTableEntry<T>::SceneDataTableEntry(SceneDataTableEntry<T>&& a_Other)
{
    memcpy(this, &a_Other, sizeof(a_Other));

    UpdateGeneratorReference();

    a_Other.m_EntryListRef = nullptr;

    m_RawData = &m_Data;
}

template <typename T>
SceneDataTableEntry<T>& SceneDataTableEntry<T>::operator=(SceneDataTableEntry<T>&& a_Other)
{
    memcpy(this, &a_Other, sizeof(a_Other));

    UpdateGeneratorReference();

    m_RawData = &m_Data;

    a_Other.m_EntryListRef = nullptr;

    return *this;
}
