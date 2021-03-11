#pragma once
#if defined(WAVEFRONT)
#include "OutputBuffer.h"
#include "MemoryBuffer.h"
#include "Camera.h"
#include "../Shaders/CppCommon/WaveFrontDataStructs.h"

#include "Renderer/LumenRenderer.h"

#include <map>
#include <string>
#include <vector>
#include <memory>

namespace WaveFront
{
    enum class LightChannel
    {
        DIRECT,
        INDIRECT,
        SPECULAR,
        NUM_CHANNELS
    };

    struct WaveFrontSettings
    {
        //The maximum path length in the scene.
        std::uint32_t depth;

        //The resolution to render at.
        glm::uvec2 renderResolution;

        //The resolution to output at (up-scaling).
        glm::uvec2 outputResolution;
    };

    class WaveFrontRenderer2WithAVengeance : public LumenRenderer
    {
        //Overridden functionality.
    public:
        std::unique_ptr<Lumen::ILumenPrimitive> CreatePrimitive(PrimitiveData& a_MeshData) override;
        std::shared_ptr<Lumen::ILumenMesh> CreateMesh(std::vector<std::unique_ptr<Lumen::ILumenPrimitive>>& a_Primitives) override;
        std::shared_ptr<Lumen::ILumenTexture> CreateTexture(void* a_PixelData, uint32_t a_Width, uint32_t a_Height) override;
        std::shared_ptr<Lumen::ILumenMaterial> CreateMaterial(const MaterialData& a_MaterialData) override;
        std::shared_ptr<Lumen::ILumenVolume> CreateVolume(const std::string& a_FilePath) override;

        //Public functionality
    public:
        WaveFrontRenderer2WithAVengeance();

        /*
         * Render.
         */
        unsigned TraceFrame();

        /*
         * Initialize the wavefront pipeline.
         * This sets up all buffers required by CUDA.
         */
        void Init(const WaveFrontSettings& a_Settings);

        //Buffers
    private:

        //OpenGL buffer to write output to.
        OutputBuffer m_OutputBuffer;

        //The surface data per pixel. 0 and 1 are used for the current and previous frame. 2 is used for any other depth.
        MemoryBuffer m_SurfaceData[3];

        //Intersection points passed to Optix.
        MemoryBuffer m_IntersectionData;

        //Buffer containing intersection rays.
        MemoryBuffer m_Rays;

        //Buffer containing shadow rays.
        MemoryBuffer m_ShadowRays;

        //Buffer used for output of separate channels of light.
        MemoryBuffer m_PixelBufferSeparate;

        //Buffer used to combine light channels after denoising.
        MemoryBuffer m_PixelBufferCombined;

        //Triangle lights.
        MemoryBuffer m_TriangleLights;

        //Variables and settings.
    private:
        WaveFrontSettings m_Settings;

        //Index of the frame, used to swap buffers.
        std::uint32_t m_FrameIndex;

        //The camera to render with.
        Camera m_Camera;
    };
}
#endif