#include "ShaderBindingTableRecord.h"

#include <algorithm>


RecordHandleBase::RecordHandleBase()
    : m_Dirty(true) // The record handles are initialized as dirty so that they are added to the shader binding table on its next update
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

void RecordHandleBase::UpdateGeneratorReference()
{
    // If both m_RecordHandleRef and m_RecordListRef are nullptrs, then this handle is invalid,
    // so it should not have any reflection on the generator
    if (m_RecordHandleRef)
    {
        *m_RecordHandleRef = this;
    }
    else if (m_RecordListRef)
    {
        (*m_RecordListRef)[m_Key] = this;
    }
};
