#pragma once
#include <map>
#include <memory>
#include <string>
#include <functional>

class CudaGLTexture;
class MemoryBuffer;

class FrameSnapshot
{
public:
    struct ImageBuffer
    {
        std::unique_ptr<CudaGLTexture> m_Memory;
        uint16_t m_Width;
        uint16_t m_Height;
    };

    FrameSnapshot(){};

    virtual ~FrameSnapshot();

    virtual void AddBuffer(std::function<std::map<std::string, ImageBuffer>()> a_ConversionLambda);

    const std::map<std::string, ImageBuffer>& GetImageBuffers() const { return m_ImageBuffers; }

private:
    std::map<std::string, ImageBuffer> m_ImageBuffers;
};

class NullFrameSnapshot : public FrameSnapshot
{
    void AddBuffer(std::function<std::map<std::string, ImageBuffer>()> a_ConversionLambda) override {}
};