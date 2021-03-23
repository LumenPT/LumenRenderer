#pragma once
#include <map>
#include <memory>
#include <string>
#include <functional>

class MemoryBuffer;

class FrameSnapshot
{
public:
    struct ImageBuffer
    {
        std::unique_ptr<MemoryBuffer> m_Memory;
        uint16_t m_Width;
        uint16_t m_Height;
    };

    FrameSnapshot(){};

    virtual ~FrameSnapshot();

    virtual void AddBuffer(std::function<std::map<std::string, ImageBuffer>()> a_ConversionLambda);

    

private:
    std::map<std::string, ImageBuffer> m_ImageBuffers;
};

class NullFrameSnapshot : public FrameSnapshot
{
    void AddBuffer(std::function<std::map<std::string, ImageBuffer>()> a_ConversionLambda) override {};
};