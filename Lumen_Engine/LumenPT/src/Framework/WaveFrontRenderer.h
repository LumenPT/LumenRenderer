#pragma once
#if defined(WAVEFRONT)
#include "OptixWrapper.h"
#include "OutputBuffer.h"
#include "MemoryBuffer.h"
#include "../Shaders/CppCommon/WaveFrontDataStructs.h"
#include "PTServiceLocator.h"

#include "Renderer/LumenRenderer.h"

#include <map>
#include <string>
#include <vector>
#include <memory>

class SceneDataTable;

namespace WaveFront
{
    struct WaveFrontSettings
    {
        //The maximum path length in the scene.
        std::uint32_t depth;

        //The resolution to render at.
        uint2 renderResolution;

        //The resolution to output at (up-scaling).
        uint2 outputResolution;

        //The minimum distance a ray has to travel before intersecting a surface.
        float minIntersectionT;

        //The maximum distance a ray can travel before terminating.
        float maxIntersectionT;
    };

    class WaveFrontRenderer : public LumenRenderer
    {
        //Overridden functionality.
    public:
        std::unique_ptr<Lumen::ILumenPrimitive> CreatePrimitive(PrimitiveData& a_PrimitiveData) override;
        std::shared_ptr<Lumen::ILumenMesh> CreateMesh(std::vector<std::unique_ptr<Lumen::ILumenPrimitive>>& a_Primitives) override;
        std::shared_ptr<Lumen::ILumenTexture> CreateTexture(void* a_PixelData, uint32_t a_Width, uint32_t a_Height) override;
        std::shared_ptr<Lumen::ILumenMaterial> CreateMaterial(const MaterialData& a_MaterialData) override;
        std::shared_ptr<Lumen::ILumenVolume> CreateVolume(const std::string& a_FilePath) override;
        std::shared_ptr<Lumen::ILumenScene> CreateScene(SceneData a_SceneData) override;

        //Public functionality
    public:
        WaveFrontRenderer();

        /*
         * Render.
         */
        unsigned TraceFrame(std::shared_ptr<Lumen::ILumenScene>& a_Scene) override;

        /*
         * Initialize the wavefront pipeline.
         * This sets up all buffers required by CUDA.
         */
        void Init(const WaveFrontSettings& a_Settings);

        //Buffers
    private:

        std::unique_ptr<MemoryBuffer> InterleaveVertexData(const PrimitiveData& a_MeshData) const;

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

        //Optix system
        std::unique_ptr<OptixWrapper> m_OptixSystem;

        //Variables and settings.
    private:
        WaveFrontSettings m_Settings;

        //Index of the frame, used to swap buffers.
        std::uint32_t m_FrameIndex;

        //The CUDA instance context stuff
        CUcontext m_CUDAContext;

        //The server locator instance.
        PTServiceLocator m_ServiceLocator;

        //Kamen's lookup table
        std::unique_ptr<SceneDataTable> m_Table;
    };
}
#endif