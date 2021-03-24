#include "FrameSnapshot.h"

#include "../Framework/MemoryBuffer.h"
#include "../Framework/CudaGLTexture.h"

FrameSnapshot::~FrameSnapshot()
{

}

void FrameSnapshot::AddBuffer(std::function<std::map<std::string, ImageBuffer>()> a_ConversionLambda)
{
    auto newBuffers = a_ConversionLambda();

    for (auto& newBuffer : newBuffers)
    {
        m_ImageBuffers[newBuffer.first] = std::move(newBuffer.second);
    }

    /*auto& buff = m_ImageBuffers[a_Name];

    buff.m_Width = a_Width;
    buff.m_Height = a_Height;

    buff.m_Memory = std::make_unique<MemoryBuffer>(a_Memory.GetSize());
    buff.m_Memory->CopyFrom(a_Memory, a_Memory.GetSize());*/

}
