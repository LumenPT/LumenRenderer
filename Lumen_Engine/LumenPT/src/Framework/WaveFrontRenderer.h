#pragma once
#if defined(WAVEFRONT)
#include "LightDataBuffer.h"
#include "OptixWrapper.h"
#include "CudaGLTexture.h"
#include "MemoryBuffer.h"
#include "GPUTexture.h"
#include "InteropGPUTexture.h"
#include "../Shaders/CppCommon/WaveFrontDataStructs.h"
#include "PTServiceLocator.h"
#include "Nvidia/INRDWrapper.h"
#include "Nvidia/IDLSSWrapper.h"
#include "../Tools/LumenPTModelConverter.h"
#include "OptixDenoiserWrapper.h" x

#include "Renderer/LumenRenderer.h"


#include <map>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <queue>

class SceneDataTable;

class FrameSnapshot;
namespace WaveFront
{
    struct WaveFrontSettings
    {

        std::filesystem::path m_ShadersFilePathSolids;

        std::filesystem::path m_ShadersFilePathVolumetrics;

        //The maximum path length in the scene.
        std::uint32_t depth;

        //The resolution to render at.
        uint2 renderResolution;

        //The resolution to output at (up-scaling).
        uint2 outputResolution;

        bool blendOutput;
    };

    class WaveFrontRenderer : public LumenRenderer
    {
        //Overridden functionality.
    public:
        void StartRendering() override;
        void PerformDeferredOperations() override;

        Lumen::SceneManager::GLTFResource OpenCustomFileFormat(const std::string& a_OriginalFilePath) override;
        Lumen::SceneManager::GLTFResource CreateCustomFileFormat(const std::string& a_OriginalFilePath) override;

        std::unique_ptr<Lumen::ILumenPrimitive> CreatePrimitive(PrimitiveData& a_PrimitiveData) override;
        std::shared_ptr<Lumen::ILumenMesh> CreateMesh(std::vector<std::shared_ptr<Lumen::ILumenPrimitive>>& a_Primitives) override;
        std::shared_ptr<Lumen::ILumenTexture> CreateTexture(void* a_PixelData, uint32_t a_Width, uint32_t a_Height, bool a_Normalize) override;
        std::shared_ptr<Lumen::ILumenMaterial> CreateMaterial(const MaterialData& a_MaterialData) override;
        std::shared_ptr<Lumen::ILumenVolume> CreateVolume(const std::string& a_FilePath) override;
        std::shared_ptr<Lumen::ILumenScene> CreateScene(SceneData a_SceneData) override;

        void InitNGX() override;

        //Public functionality
    public:
        WaveFrontRenderer();

        ~WaveFrontRenderer();

        /*
         * Render.
         */
        unsigned GetOutputTexture() override;

        std::vector<uint8_t> GetOutputTexturePixels(uint32_t& a_Width, uint32_t& a_Height) override;

        /*
         * Initialize the wavefront pipeline.
         * This sets up all buffers required by CUDA.
         */
        void Init(const WaveFrontSettings& a_Settings);

        void BeginSnapshot() override;

        std::unique_ptr<FrameSnapshot> EndSnapshot() override;

        void SetRenderResolution(glm::uvec2 a_NewResolution) override;
        void SetOutputResolution(glm::uvec2 a_NewResolution) override;
        glm::uvec2 GetRenderResolution() override;
        glm::uvec2 GetOutputResolution() override;

        /*
         * Set the append mode.
         * When true, final output is blended and not overwritten to build a higher res image over time.
         * When false, output is overwritten every frame.
         */
        void SetBlendMode(bool a_Blend) override;

        /*
         * Get the append mode.
         * When true, output is blended and not overwritten.
         */
        bool GetBlendMode() const override;


        struct DenoiserSettings
        {
            DenoiserSettings()
                : m_UseOptix(false)
                , m_UseNRD(false)
                , m_OptixAlbedo(true)
                , m_OptixNormal(true)
                , m_OptixTemporal(false)
            {}
            bool m_UseOptix;
            bool m_UseNRD;

            bool m_OptixAlbedo;
            bool m_OptixNormal;
            bool m_OptixTemporal;
        } m_DenoiserSettings;
    	
    private:

        void TraceFrame();

        std::unique_ptr<MemoryBuffer> InterleaveVertexData(const PrimitiveData& a_MeshData) const;

        void ResizeBuffers();

        void SetOutputResolutionInternal(glm::uvec2 a_NewResolution);
        void WaitForDeferredCalls(); // Stalls the thread until all calls that are deferred to other treads are performed

        void FinalizeFrameStats();

        //OpenGL buffer to write output to.
        std::unique_ptr<CudaGLTexture> m_OutputBuffer;

        // Intermediate buffer to store path tracing results without interfering with OpenGL rendering thread
        MemoryBuffer m_IntermediateOutputBuffer;

        //The surface data per pixel. 0 and 1 are used for the current and previous frame. 2 is used for any other depth.
        MemoryBuffer m_SurfaceData[3];

        //The volumetric data per pixel.
        MemoryBuffer m_VolumetricData[1];

        //Intersection points passed to Optix.
        MemoryBuffer m_IntersectionData;

		//Volumetric intersection data passed to Optix
		MemoryBuffer m_VolumetricIntersectionData;

        //Buffer containing intersection rays.
        MemoryBuffer m_Rays;

        //Buffer containing shadow rays.
        MemoryBuffer m_ShadowRays;

        //Buffer containing volumetric shadow rays.
        MemoryBuffer m_VolumetricShadowRays;

        static const unsigned s_numLightChannels = static_cast<unsigned>(LightChannel::NUM_CHANNELS);
        //Buffer used for output of separate channels of light.
        std::unique_ptr<InteropGPUTexture> m_PixelBufferSeparate[s_numLightChannels];

        //Buffer used to combine light channels after denoising.
        std::unique_ptr<InteropGPUTexture> m_PixelBufferCombined;
        
        std::unique_ptr<InteropGPUTexture> m_PixelBufferUpscaled;

        std::unique_ptr<InteropGPUTexture> m_DepthBuffer;
        
        std::unique_ptr<InteropGPUTexture> m_JitterBuffer;

        //Buffer containing motion vectors
        std::unique_ptr<InteropGPUTexture> m_MotionVectorBuffer;

        //Buffer containing Normal(world-space,XYZ) and Roughness.
        std::unique_ptr<InteropGPUTexture> m_NormalRoughnessBuffer;
        
        Microsoft::WRL::ComPtr<ID3D11Texture2D> m_D3D11JitterBuffer;

        Microsoft::WRL::ComPtr<ID3D11Texture2D> m_D3D11PixelBufferSeparate;

        Microsoft::WRL::ComPtr<ID3D11Texture2D> m_D3D11PixelBufferCombined;
        
        Microsoft::WRL::ComPtr<ID3D11Texture2D> m_D3D11PixelBufferUpscaled;

        Microsoft::WRL::ComPtr<ID3D11Texture2D> m_D3D11DepthBuffer;

        Microsoft::WRL::ComPtr<ID3D11Texture2D> m_D3D11MotionVectorBuffer;

        Microsoft::WRL::ComPtr<ID3D11Texture2D> m_D3D11NormalRoughnessBuffer;

        Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_D3D11PixelBufferCombinedUAV;
        Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_D3D11PixelBufferUpscaledUAV;

        //Triangle lights.
        std::unique_ptr<LightDataBuffer> m_LightDataBuffer;

        //Optix system
        std::unique_ptr<OptixWrapper> m_OptixSystem;

        //DX11System
        std::unique_ptr<DX11Wrapper> m_DX11Wrapper;

        //NRI Wrapper
        std::unique_ptr<INRDWrapper> m_NRD;

        //DLSS Wrapper
        std::unique_ptr<IDLSSWrapper> m_DLSS;

        std::unique_ptr<OptixDenoiserWrapper> m_OptixDenoiser;

        //ReSTIR
        std::unique_ptr<ReSTIR> m_ReSTIR;

        //Variables and settings.
    private:

        inline void ResizeInteropTexture(
            const std::unique_ptr<InteropGPUTexture>& a_InteropTexture,
            Microsoft::WRL::ComPtr<ID3D11Texture2D>& a_TextureResource,
            const uint3& a_NewSize) const;


        WaveFrontSettings m_Settings; // Settings to use while rendering
        WaveFrontSettings m_IntermediateSettings; // Settings used to make changes from other threads without affecting the rendering process
        std::mutex m_SettingsUpdateMutex;

        std::queue<std::function<void()>> m_DeferredOpenGLCalls;
        std::condition_variable m_OGLCallCondition;

        //Blend counter when blending is enabled.
        unsigned m_BlendCounter;

        //Index of the frame, used to swap buffers.
        std::uint32_t m_FrameIndex;

        //The CUDA instance context stuff
        CUcontext m_CUDAContext;

        //The server locator instance.
        PTServiceLocator m_ServiceLocator;

        std::mutex m_OutputBufferMutex;
        std::thread m_PathTracingThread;
        std::condition_variable m_OutputCondition;
        class GLFWwindow* m_GLContext;
        bool m_StopRendering;

        GLuint m_OutputTexture;

        // The Frame Snapshot is used to define what to record when the output layer requests that
        // See TraceFrame() ##ToolsBookmark for example
        std::unique_ptr<FrameSnapshot> m_FrameSnapshot;
        bool m_SnapshotReady;
        bool m_StartSnapshot;

        LumenPTModelConverter m_ModelConverter;

        FrameStats m_CurrentFrameStats;
    };
}
#endif