#pragma once

#include "../Shaders/CppCommon/LaunchParameters.h"

#include <unordered_map>

template<typename T>
struct Record
{
    ProgramGroupHeader m_Header;
    T m_Data;
};

template<>
struct Record<void>
{
    ProgramGroupHeader m_Header;
};

class RecordHandleBase
{
    friend class ShaderBindingTableGenerator;
public:

    RecordHandleBase();
        
    virtual ~RecordHandleBase();
    
    RecordHandleBase(RecordHandleBase&) = delete;
    RecordHandleBase& operator=(RecordHandleBase&) = delete;

    uint32_t m_TableIndex;

protected:
    void UpdateGeneratorReference();

    void* m_RawData; // Pointer to the record in the child object
    uint32_t m_Size; // Size of the record owned by the child object

    std::unordered_map<uint64_t, RecordHandleBase*>* m_RecordListRef;
    RecordHandleBase** m_RecordHandleRef;

    uint64_t m_Key;
    bool m_Dirty;
    uint32_t count;

};

template<typename T>
class RecordHandle : public RecordHandleBase
{
public:
    RecordHandle()
        : RecordHandleBase()
    {
        m_Size = sizeof(Record<T>);
        m_RawData = &m_Record;
    }

    ~RecordHandle();

    RecordHandle(RecordHandle<T>&& a_Other);

    RecordHandle& operator=(RecordHandle<T>&& a_Other);        

    Record<T>& GetRecord()
    {
        //m_RawData = &m_Record;
        m_Dirty = true;
        return m_Record;
    };

    const Record<T>& GetConstRecord() const
    {
        return m_Record;
    };
private:
    Record<T> m_Record;
};

template <typename T>
RecordHandle<T>::~RecordHandle()
{
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

    a_Other.m_RecordHandleRef = nullptr;
    a_Other.m_RecordListRef = nullptr;

    m_RawData = &m_Record;
}

template <typename T>
RecordHandle<T>& RecordHandle<T>::operator=(RecordHandle<T>&& a_Other)
{
    //RecordHandle<T> newHandle;
    memcpy(this, &a_Other, sizeof(a_Other));

    UpdateGeneratorReference();

    m_RawData = &m_Record;

    a_Other.m_RecordHandleRef = nullptr;
    a_Other.m_RecordListRef = nullptr;

    return *this;
}
