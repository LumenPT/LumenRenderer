#pragma once

#include "../Shaders/CppCommon/LaunchParameters.h"

#include <unordered_map>
static int counter = 9;


template<typename T>
struct Record
{
    ProgramGroupHeader m_Header;
    T m_Data;
};

class RecordHandleBase
{
    friend class ShaderBindingTableGenerator;
public:

    RecordHandleBase();
        
    virtual ~RecordHandleBase();
    
    RecordHandleBase(RecordHandleBase&) = delete;
    RecordHandleBase& operator=(RecordHandleBase&) = delete;

    //RecordHandleBase(RecordHandleBase&& a_Other);
    //RecordHandleBase& operator=(RecordHandleBase&& a_Other);

    void debug()
    {
        if (count == 18)
        {
            printf("asd");
        }
    }

    uint32_t m_TableIndex;

protected:
    void UpdateGeneratorReference();

    void* m_RawData;
    uint32_t m_Size;

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
        debug();
        count = counter++;
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
    printf("Destr\n");

    if (m_RecordListRef)
        m_RecordListRef->erase(m_Key);
    else if (m_RecordHandleRef)
        *m_RecordHandleRef = nullptr;
}

template <typename T>
RecordHandle<T>::RecordHandle(RecordHandle<T>&& a_Other)
{
    count = counter++;
    memcpy(this, &a_Other, sizeof(a_Other));

    debug();
    a_Other.debug();

    UpdateGeneratorReference();

    a_Other.m_RecordHandleRef = nullptr;
    a_Other.m_RecordListRef = nullptr;
    printf("M C D\n");

    m_RawData = &m_Record;
    count = counter++;
    debug();
}

template <typename T>
RecordHandle<T>& RecordHandle<T>::operator=(RecordHandle<T>&& a_Other)
{
    count = counter++;
    //RecordHandle<T> newHandle;
    memcpy(this, &a_Other, sizeof(a_Other));

    debug();
    a_Other.debug();

    UpdateGeneratorReference();

    m_RawData = &m_Record;

    a_Other.m_RecordHandleRef = nullptr;
    a_Other.m_RecordListRef = nullptr;

    printf("M A D\n");

    return *this;
}
