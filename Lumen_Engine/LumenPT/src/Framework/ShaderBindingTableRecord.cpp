#include "ShaderBindingTableRecord.h"

#include <algorithm>


RecordHandleBase::RecordHandleBase()
    : m_Dirty(true)
    , m_TableIndex(0)
    , m_Size(0)
    , m_RawData(nullptr)
    , m_RecordListRef(nullptr)
    , m_Key(0)
    , m_RecordHandleRef(nullptr)
{};

RecordHandleBase::~RecordHandleBase()
{
   
}

//RecordHandleBase::RecordHandleBase(RecordHandleBase&& a_Other)
//{
//    // Big brain play here. m_Size is the size of the child instantiated class. Copies the exact data contained by the child class.
//    memcpy(this, &a_Other, sizeof(RecordHandleBase) + a_Other.m_Size);
//
//    UpdateGeneratorReference();
//
//    a_Other.m_RecordListRef = nullptr;
//    a_Other.m_RecordHandleRef = nullptr;
//}

void RecordHandleBase::UpdateGeneratorReference()
{
    if (m_RecordHandleRef)
    {
        *m_RecordHandleRef = this;
    }
    else if (m_RecordListRef)
    {
        (*m_RecordListRef)[m_Key] = this;
    }
};
