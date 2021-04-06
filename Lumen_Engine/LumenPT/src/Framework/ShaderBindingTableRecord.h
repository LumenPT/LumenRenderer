#pragma once

#include "../Shaders/CppCommon/LaunchParameters.h"

#include <unordered_map>

// A record is a ProgramGroupHeader with additional data attached to it
// The type of the data is specified via the template parameter
template<typename T>
struct Record
{
    ProgramGroupHeader m_Header;
    T m_Data;
};

// Template specialization to remove m_Data if it would be void
template<>
struct Record<void>
{
    ProgramGroupHeader m_Header;
};

// The base record handle class is used as means of having a common way to store all different record handle types
class RecordHandleBase
{
    friend class ShaderBindingTableGenerator;
public:

    RecordHandleBase();
    
    virtual ~RecordHandleBase();

    // The record handles cannot be copied, only moved
    RecordHandleBase(RecordHandleBase&) = delete;
    RecordHandleBase& operator=(RecordHandleBase&) = delete;

    // The table index dictates what position the record has in the shader binding table
    uint32_t m_TableIndex;

protected:
    // Update the generator's pointer to point to this when a move is performed
    void UpdateGeneratorReference();

    void* m_RawData; // Pointer to the record in the child object
    uint32_t m_Size; // Size of the record owned by the child object

    // Reference to the list of records which has this record referenced. Used for miss and hit group records
    std::unordered_map<uint64_t, RecordHandleBase*>* m_RecordListRef;
    // Reference to the reference used by the generator. Used for the ray gen records
    RecordHandleBase** m_RecordHandleRef;

    uint64_t m_Key; // Key used to find the record in the generator's list
    bool m_Dirty; // Has the record been modified since the last time the shader binding table has been updated
};

template<typename T>
class RecordHandle : public RecordHandleBase
{
public:
    RecordHandle()
        : RecordHandleBase()
    {
        // The generator only cares where the data is located and big it is
        // Because of this, the child class saves the size of the data and where it stores it in the parent class' variables
        m_Size = sizeof(Record<T>);
        m_RawData = &m_Record;
    }

    ~RecordHandle();

    // RecordHandles cannot be copied, but can be moved in a specific way
    RecordHandle(RecordHandle<T>&& a_Other);
    RecordHandle& operator=(RecordHandle<T>&& a_Other);        

    // Returns a writable record reference. Changes to it are reflected in the table. Marks the handle as dirty.
    // Use GetConstRecord() if you only want to read from the record.
    Record<T>& GetRecord()
    {
        //m_RawData = &m_Record;
        m_Dirty = true;
        return m_Record;
    };

    // Returns a read-only record
    const Record<T>& GetConstRecord() const
    {
        return m_Record;
    };
private:
    // The record that is handled by the handle.
    Record<T> m_Record;
};

template <typename T>
RecordHandle<T>::~RecordHandle()
{
    // The movement assignment operator and constructor use the record list reference and recond handle reference
    // to track if the handle is valid or not.
    // If this handle is invalidated, we do not want its destruction to be reflected in the generator
    if (m_RecordListRef)
        m_RecordListRef->erase(m_Key);
    else if (m_RecordHandleRef)
        *m_RecordHandleRef = nullptr;
}

template <typename T>
RecordHandle<T>::RecordHandle(RecordHandle<T>&& a_Other)
{
    memcpy(this, &a_Other, sizeof(a_Other));

    UpdateGeneratorReference();

    // Invalidate the original record handle
    a_Other.m_RecordHandleRef = nullptr;
    a_Other.m_RecordListRef = nullptr;

    // The raw data pointer needs to be updated because the record handle has just moved to a new location in memory,
    // so the old pointer is now pointing to unused memory
    m_RawData = &m_Record;
}

template <typename T>
RecordHandle<T>& RecordHandle<T>::operator=(RecordHandle<T>&& a_Other)
{
    //RecordHandle<T> newHandle;
    memcpy(this, &a_Other, sizeof(a_Other));

    UpdateGeneratorReference();

    // Invalidate the original record handle
    a_Other.m_RecordHandleRef = nullptr;
    a_Other.m_RecordListRef = nullptr;

    // The raw data pointer needs to be updated because the record handle has just moved to a new location in memory,
    // so the old pointer is now pointing to unused memory
    m_RawData = &m_Record;

    return *this;
}
