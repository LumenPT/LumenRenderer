#pragma once

#include <unordered_map>
#include <set>

// The base class is necessary to have a common way of storing all entries in the generator class,
// Despite them all having different data attached to them
class SceneDataTableEntryBase
{
    friend class SceneDataTable;
public:

    SceneDataTableEntryBase();

    virtual ~SceneDataTableEntryBase();

    // The entry handles cannot be copied in any way, only moved
    SceneDataTableEntryBase(SceneDataTableEntryBase&) = delete;
    SceneDataTableEntryBase& operator=(SceneDataTableEntryBase&) = delete;

    // The index that is used to access this entry within the scene data table
    uint32_t m_TableIndex;
    bool m_TableIndexValid;

protected:
    // Update the pointer within the generator to point to a new entry in the case of movement
    void UpdateGeneratorReference();

    void* m_RawData; // Pointer to the record in the child object
    uint32_t m_Size; // Size of the record owned by the child object

    // Reference to the list containing the entry
    std::unordered_map<uint64_t, SceneDataTableEntryBase*>* m_EntryListRef;
    std::set<uint64_t>* m_KeyListRef;

    uint64_t m_Key; // Key used to find the entry in the list
    bool m_Dirty; // Has the entry been modified since last time the table was updated
};

// The derived class is templated to allow for different struct types to be used as entries
template<typename T>
class SceneDataTableEntry : public SceneDataTableEntryBase
{
public:
    SceneDataTableEntry()
        : SceneDataTableEntryBase()
    {
        // The generator only wants to know how big the data is, and where it is located.
        // Because of this they are saved here as pointers
        m_Size = sizeof(T);
        m_RawData = &m_Data;
    }

    ~SceneDataTableEntry();

    // The class cannot be copied, but can be moved in a specific way.
    SceneDataTableEntry(SceneDataTableEntry<T>&& a_Other);
    SceneDataTableEntry& operator=(SceneDataTableEntry<T>&& a_Other);

    // Get the data for write. Marks the entry as dirty. Use GetConstData() if you only want to read from it.
    T& GetData()
    {
        m_Dirty = true;
        return m_Data;
    };

    // Returns the data of the entry in a read-only form.
    const T& GetConstData() const
    {
        return m_Data;
    };
private:
    // The data carried by the entry
    T m_Data;
};

template <typename T>
SceneDataTableEntry<T>::~SceneDataTableEntry()
{
    // The movement assignment operator and movement constructor invalidate the original instance
    // by setting the entry list reference to a nullptr.
    // This is used to determine if this is the real instance that is being deleted
    if (m_EntryListRef)
        m_EntryListRef->erase(m_Key);
    if (m_KeyListRef)
        m_KeyListRef->emplace(m_Key);
        
}

template <typename T>
SceneDataTableEntry<T>::SceneDataTableEntry(SceneDataTableEntry<T>&& a_Other)
{
    memcpy(this, &a_Other, sizeof(a_Other));

    UpdateGeneratorReference();

    // Invalidate the original entry so that its destruction is not reflected in the generator
    a_Other.m_EntryListRef = nullptr;
    a_Other.m_KeyListRef = nullptr;

    // The raw data pointer is changed because the entry is allocated at a new place in memory
    m_RawData = &m_Data;
}

template <typename T>
SceneDataTableEntry<T>& SceneDataTableEntry<T>::operator=(SceneDataTableEntry<T>&& a_Other)
{
    memcpy(this, &a_Other, sizeof(a_Other));

    UpdateGeneratorReference();

    // Invalidate the original entry so that its destruction is not reflected in the generator
    a_Other.m_EntryListRef = nullptr;
    a_Other.m_KeyListRef = nullptr;

    // The raw data pointer is changed because the entry is allocated at a new place in memory
    m_RawData = &m_Data;

    return *this;
}
