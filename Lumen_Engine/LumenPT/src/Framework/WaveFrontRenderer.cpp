#include "../Shaders/CppCommon/RenderingUtility.h"
#include "Timer.h"
#if defined(WAVEFRONT)

#include "WaveFrontRenderer.h"
#include "PTPrimitive.h"
#include "PTMesh.h"
#include "PTScene.h"
#include "PTMaterial.h"
#include "PTTexture.h"
#include "PTVolume.h"
#include "PTMaterial.h"
#include "MemoryBuffer.h"
#include "CudaGLTexture.h"
#include "SceneDataTable.h"
#include "../CUDAKernels/WaveFrontKernels.cuh"
#include "../Shaders/CppCommon/WaveFrontDataStructs.h"
#include "CudaUtilities.h"
#include "ReSTIR.h"
#include "../Tools/FrameSnapshot.h"
#include "../Tools/SnapShotProcessing.cuh"
//#include "Lumen/Window.h"
#include "Lumen/LumenApp.h"
#include "DX11Wrapper.h"

//#include "LumenPTConfig.h"

#ifdef USE_NVIDIA_DENOISER
#include "Nvidia/NRDWrapper.h"
using NrdWrapper = NRDWrapper;
#else
#include "Nvidia/NullNRDWrapper.h"
using NrdWrapper = NullNRDWrapper;
#endif

#ifdef USE_NVIDIA_DLSS
#include "Nvidia/DLSSWrapper.h"
using DlssWrapper = DLSSWrapper;
#else
#include "Nvidia/NullDLSSWrapper.h"
using DlssWrapper = NullDLSSWrapper;
#endif

#include "../../../Lumen/vendor/GLFW/include/GLFW/glfw3.h"
#include <Optix/optix_function_table_definition.h>
#include <filesystem>
#include <glm/gtx/compatibility.hpp>
#include <sutil/Matrix.h>
#include "../Framework/PTMeshInstance.h"



sutil::Matrix4x4 ConvertGLMtoSutilMat4(const glm::mat4& glmMat)
{
    float data[16];
	
    for (int row = 0; row < 4; ++row)
    {
        for (int column = 0; column < 4; ++column)
        {
            data[row * 4 + column] = glmMat[column][row]; //we swap column and row indices because sutil is in row major while glm is in column major
        }
    }

    return sutil::Matrix4x4(data);
}

namespace WaveFront
{
    void WaveFrontRenderer::Init(const WaveFrontSettings& a_Settings)
    {
        m_DX11Wrapper = std::make_unique<DX11Wrapper>();
        m_ServiceLocator.m_DX11Wrapper = m_DX11Wrapper.get();
        m_DX11Wrapper->Init();

        m_BlendCounter = 0;
        m_FrameIndex = 0;
        m_Settings = a_Settings;

        // The intermediate settings are used to get the values to display via ImGui
        // Thus initializing them to the same values as the real settings will ensure ImGui displays the correct data
        m_IntermediateSettings = m_Settings;

        //Init CUDA
        cudaFree(0);
        m_CUDAContext = 0;

        //TODO: Ensure shader names match what we put down here.
        OptixWrapper::InitializationData optixInitData;
        optixInitData.m_CUDAContext = m_CUDAContext;
		optixInitData.m_SolidProgramData.m_ProgramPath = a_Settings.m_ShadersFilePathSolids;
        optixInitData.m_SolidProgramData.m_ProgramLaunchParamName = "launchParams";
        optixInitData.m_SolidProgramData.m_ProgramRayGenFuncName = "__raygen__WaveFrontRG";
        optixInitData.m_SolidProgramData.m_ProgramMissFuncName = "__miss__WaveFrontMS";
        optixInitData.m_SolidProgramData.m_ProgramAnyHitFuncName = "__anyhit__WaveFrontAH";
        optixInitData.m_SolidProgramData.m_ProgramClosestHitFuncName = "__closesthit__WaveFrontCH";
        optixInitData.m_VolumetricProgramData.m_ProgramPath = a_Settings.m_ShadersFilePathVolumetrics;
        optixInitData.m_VolumetricProgramData.m_ProgramIntersectionFuncName = "__intersection__Volumetric";
        optixInitData.m_VolumetricProgramData.m_ProgramAnyHitFuncName = "__anyhit__Volumetric";
        optixInitData.m_VolumetricProgramData.m_ProgramClosestHitFuncName = "__closesthit__Volumetric";
        optixInitData.m_PipelineMaxNumHitResultAttributes = 2;
        optixInitData.m_PipelineMaxNumPayloads = 5;

        

        m_OptixSystem = std::make_unique<OptixWrapper>(optixInitData);

        //Set the service locator's pointer to the OptixWrapper.
        m_ServiceLocator.m_OptixWrapper = m_OptixSystem.get();

        /*m_NRD = std::make_unique<NrdWrapper>();
        NRDWrapperInitParams nrdInitParams;
        nrdInitParams.m_InputImageWidth = m_Settings.renderResolution.x;
        nrdInitParams.m_InputImageHeight = m_Settings.renderResolution.y;
        nrdInitParams.m_pServiceLocator = &m_ServiceLocator;
        m_NRD->Initialize(nrdInitParams);*/

        m_OptixDenoiser = std::make_unique<OptixDenoiserWrapper>();

        //Set up the OpenGL output buffer.
        m_OutputBuffer = std::make_unique<CudaGLTexture>(GL_RGBA8, m_Settings.outputResolution.x, m_Settings.outputResolution.y, 4);
        SetRenderResolution(glm::uvec2(m_Settings.renderResolution.x, m_Settings.renderResolution.y));
        SetOutputResolutionInternal(glm::uvec2(m_Settings.outputResolution.x, m_Settings.outputResolution.y));

        //Set up buffers.

        //Set up pixel output buffers.

        //Create d3d11 texture2D which will be used for the pixel-buffer-separate containing the different light channels.
        m_D3D11PixelBufferSeparate = m_DX11Wrapper->CreateTexture2D(
            { m_Settings.renderResolution.x, m_Settings.renderResolution.y, s_numLightChannels},
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            D3D11_BIND_FLAG::D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_FLAG::D3D11_BIND_SHADER_RESOURCE
            );

        //Create d3d11 texture2D which will be used for the pixel-buffer-combined containing the merged light channels.
        m_D3D11PixelBufferCombined = m_DX11Wrapper->CreateTexture2D(
            { m_Settings.renderResolution.x, m_Settings.renderResolution.y, 1 }, 
            DXGI_FORMAT_R16G16B16A16_FLOAT, 
            D3D11_BIND_FLAG::D3D11_BIND_SHADER_RESOURCE);
        
        //Create d3d11 texture2D which will be an upscaled version of the m_D3D11PixelBufferCombined done by DLSS
        m_D3D11PixelBufferUpscaled = m_DX11Wrapper->CreateTexture2D(
            { m_Settings.outputResolution.x, m_Settings.outputResolution.y, 1 }, 
            DXGI_FORMAT_R16G16B16A16_FLOAT, 
            D3D11_BIND_FLAG::D3D11_BIND_UNORDERED_ACCESS);

        // Create Unordered Access Views for Pixelbuffer Combined and Upscaled for use by DLSS 
        D3D11_UNORDERED_ACCESS_VIEW_DESC desc = {};
        desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        desc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
        //m_DX11Wrapper->CreateUAV(m_D3D11PixelBufferCombined, &desc, m_D3D11PixelBufferCombinedUAV);
        m_DX11Wrapper->CreateUAV(m_D3D11PixelBufferUpscaled, &desc, m_D3D11PixelBufferUpscaledUAV);

        //Create d3d11 texture2D which will be used for the depth buffer containing the depth values for each pixel.
        m_D3D11DepthBuffer = m_DX11Wrapper->CreateTexture2D(
            {m_Settings.renderResolution.x, m_Settings.renderResolution.y, 1}, 
            DXGI_FORMAT_R32_FLOAT,
            D3D11_BIND_FLAG::D3D11_BIND_SHADER_RESOURCE);

        m_D3D11JitterBuffer = m_DX11Wrapper->CreateTexture2D(
            { m_Settings.renderResolution.x, m_Settings.renderResolution.y, 1 }, 
            DXGI_FORMAT_R16G16_FLOAT,
            D3D11_BIND_FLAG::D3D11_BIND_SHADER_RESOURCE);

        //Create d3d11 texture2D which will be used for the motion vector buffer containing the motion vectors for each pixel.
        m_D3D11MotionVectorBuffer = m_DX11Wrapper->CreateTexture2D(
            { m_Settings.renderResolution.x, m_Settings.renderResolution.y, 1 },
            DXGI_FORMAT_R16G16_FLOAT,
            D3D11_BIND_FLAG::D3D11_BIND_SHADER_RESOURCE);

        //Create d3d11 texture2D which will be used for the normal-roughness buffer containing the normal and rougness for each pixel.
        m_D3D11NormalRoughnessBuffer = m_DX11Wrapper->CreateTexture2D(
            { m_Settings.renderResolution.x, m_Settings.renderResolution.y, 1 }, 
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            D3D11_BIND_FLAG::D3D11_BIND_SHADER_RESOURCE);

        {//Create interop-texture for pixel-buffers.
            //Description of the cuda texture created with an interop-texture for each pixel buffer.
            cudaTextureDesc pixelBufferDesc{};
            memset(&pixelBufferDesc, 0, sizeof(pixelBufferDesc));

            pixelBufferDesc.addressMode[0] = cudaAddressModeClamp;
            pixelBufferDesc.addressMode[1] = cudaAddressModeClamp;
            pixelBufferDesc.filterMode = cudaFilterModePoint;
            pixelBufferDesc.readMode = cudaReadModeElementType;

            //Create interop-textures for each light channel contained in the pixel-buffer-separate.
            for (unsigned int channelIndex = 0; channelIndex < s_numLightChannels; ++channelIndex)
            {
                m_PixelBufferSeparate[channelIndex] = std::make_unique<InteropGPUTexture>(
                    m_D3D11PixelBufferSeparate, 
                    pixelBufferDesc, 
                    cudaGraphicsRegisterFlagsSurfaceLoadStore);

                //Make sure to initialize the buffer with 0s
                m_PixelBufferSeparate[channelIndex]->Map(channelIndex);
                m_PixelBufferSeparate[channelIndex]->Clear();
                m_PixelBufferSeparate[channelIndex]->Unmap();
            }

            //Create interop-texture for the single light channel contained in the pixel-buffer-combined.
            m_PixelBufferCombined = std::make_unique<InteropGPUTexture>(
                m_D3D11PixelBufferCombined, 
                pixelBufferDesc, 
                cudaGraphicsRegisterFlagsSurfaceLoadStore);

            //Make sure to initialize the buffer with 0s
            m_PixelBufferCombined->Map();
            m_PixelBufferCombined->Clear();
            m_PixelBufferCombined->Unmap();

            //This has to go in WriteToOutput
            m_PixelBufferUpscaled = std::make_unique<InteropGPUTexture>(
                m_D3D11PixelBufferUpscaled,
                pixelBufferDesc,
                cudaGraphicsRegisterFlagsSurfaceLoadStore);

            m_PixelBufferUpscaled->Map();
            m_PixelBufferUpscaled->Clear();
            m_PixelBufferUpscaled->Unmap();
        }

        {//Create interop-texture for the depth buffer.
            cudaTextureDesc depthBufferDesc{};
            memset(&depthBufferDesc, 0, sizeof(depthBufferDesc));

            depthBufferDesc.addressMode[0] = cudaAddressModeClamp;
            depthBufferDesc.addressMode[1] = cudaAddressModeClamp;
            depthBufferDesc.filterMode = cudaFilterModeLinear;
            depthBufferDesc.readMode = cudaReadModeElementType;

            m_DepthBuffer = std::make_unique<InteropGPUTexture>(
                m_D3D11DepthBuffer, 
                depthBufferDesc, 
                cudaGraphicsRegisterFlagsSurfaceLoadStore);

            //Make sure the depth-buffer is initialized with 0s
            m_DepthBuffer->Map();
            m_DepthBuffer->Clear();
            m_DepthBuffer->Unmap();

            // see how it goes with depth buffer description 
            m_JitterBuffer = std::make_unique<InteropGPUTexture>(
                m_D3D11JitterBuffer, 
                depthBufferDesc, 
                cudaGraphicsRegisterFlagsSurfaceLoadStore);

            m_JitterBuffer->Map();
            m_JitterBuffer->Clear();
            m_JitterBuffer->Unmap();
        }

        //Create interop-texture for the depth buffer.
        cudaTextureDesc depthBufferDesc{};
        memset(&depthBufferDesc, 0, sizeof(depthBufferDesc));

        { //Create interop-texture for the motion vector buffer.
            cudaTextureDesc motionVectorBufferDesc{};
            memset(&motionVectorBufferDesc, 0, sizeof(motionVectorBufferDesc));

            motionVectorBufferDesc.addressMode[0] = cudaAddressModeClamp;
            motionVectorBufferDesc.addressMode[1] = cudaAddressModeClamp;
            motionVectorBufferDesc.filterMode = cudaFilterModePoint;
            motionVectorBufferDesc.readMode = cudaReadModeElementType;

            m_MotionVectorBuffer = std::make_unique<InteropGPUTexture>(
                m_D3D11MotionVectorBuffer, 
                motionVectorBufferDesc, 
                cudaGraphicsRegisterFlagsSurfaceLoadStore);

            m_MotionVectorBuffer->Map();
            m_MotionVectorBuffer->Clear();
            m_MotionVectorBuffer->Unmap();
        }

        {//Create interop-texture for the normal-rougness texture.
            cudaTextureDesc normalRougnessBufferDesc{};
            memset(&normalRougnessBufferDesc, 0, sizeof(normalRougnessBufferDesc));

            normalRougnessBufferDesc.addressMode[0] = cudaAddressModeClamp;
            normalRougnessBufferDesc.addressMode[0] = cudaAddressModeClamp;
            normalRougnessBufferDesc.filterMode = cudaFilterModePoint;
            normalRougnessBufferDesc.readMode = cudaReadModeElementType;

            m_NormalRoughnessBuffer = std::make_unique<InteropGPUTexture>(
                    m_D3D11NormalRoughnessBuffer, 
                    normalRougnessBufferDesc, 
                    cudaGraphicsRegisterFlagsSurfaceLoadStore);

        }

        ResizeBuffers();

        m_LightDataBuffer = std::make_unique<LightDataBuffer>(1000000);

        //Set the service locator pointer to point to the m'table.
        /*m_Table = std::make_unique<SceneDataTable>();
        m_ServiceLocator.m_SceneDataTable = m_Table.get();*/
        CHECKLASTCUDAERROR;

        m_ServiceLocator.m_Renderer = this;
        m_FrameSnapshot = std::make_unique<NullFrameSnapshot>();

        m_ModelConverter.SetRendererRef(*this);

        m_NRD = std::make_unique<NrdWrapper>();
        NRDWrapperInitParams nrdInitParams;
        nrdInitParams.m_InputImageWidth = m_Settings.renderResolution.x;
        nrdInitParams.m_InputImageHeight = m_Settings.renderResolution.y;
        nrdInitParams.m_pServiceLocator = &m_ServiceLocator;
        nrdInitParams.m_InputDiffTex = m_D3D11PixelBufferSeparate.Get();
        nrdInitParams.m_InputSpecTex = m_D3D11PixelBufferSeparate.Get();
        nrdInitParams.m_NormalRoughnessTex = m_D3D11NormalRoughnessBuffer.Get();
        nrdInitParams.m_MotionVectorTex = m_D3D11MotionVectorBuffer.Get();
        nrdInitParams.m_ViewZTex = m_D3D11DepthBuffer.Get();
        nrdInitParams.m_OutputDiffTex = m_D3D11PixelBufferSeparate.Get();
        nrdInitParams.m_OutputSpecTex = m_D3D11PixelBufferSeparate.Get();
        m_NRD->Initialize(nrdInitParams);

        m_CurrentFrameStats.m_Id = 0;
    }

    void WaveFrontRenderer::BeginSnapshot()
    {
        m_StartSnapshot = true;
    }

    std::unique_ptr<FrameSnapshot> WaveFrontRenderer::EndSnapshot()
    {
        if (m_SnapshotReady)
        {
            // Move the snapshot to a temporary variable to return shortly
            auto snap = std::move(m_FrameSnapshot);
            // Make the snapshot a Null once again to stop recording
            m_FrameSnapshot = std::make_unique<NullFrameSnapshot>();
            m_SnapshotReady = false;
            return std::move(snap);
        }
        return nullptr;
    }

    void WaveFrontRenderer::SetRenderResolution(glm::uvec2 a_NewResolution)
    {

        std::unique_lock lock(m_SettingsUpdateMutex);

        m_IntermediateSettings.renderResolution.x = a_NewResolution.x;
        m_IntermediateSettings.renderResolution.y = a_NewResolution.y;

        // Call the internal version of this function which does not involve mutex locking
        SetOutputResolutionInternal(a_NewResolution); //Separate output res, or tie it to render res based on DLSS 
    }


    void WaveFrontRenderer::SetOutputResolution(glm::uvec2 a_NewResolution)
    {
        std::lock_guard lock(m_SettingsUpdateMutex);
        m_IntermediateSettings.outputResolution.x = a_NewResolution.x;
        m_IntermediateSettings.outputResolution.y = a_NewResolution.y;
    }

    glm::uvec2 WaveFrontRenderer::GetRenderResolution()
    {

        return glm::uvec2(m_IntermediateSettings.renderResolution.x, m_IntermediateSettings.renderResolution.y);

    }

    glm::uvec2 WaveFrontRenderer::GetOutputResolution()
    {

        return glm::uvec2(m_IntermediateSettings.outputResolution.x, m_IntermediateSettings.outputResolution.y);

    }

    void WaveFrontRenderer::SetBlendMode(bool a_Blend)
    {
        m_IntermediateSettings.blendOutput = a_Blend;
        if (a_Blend) m_BlendCounter = 0;
    }

    bool WaveFrontRenderer::GetBlendMode() const
    {
        return m_IntermediateSettings.blendOutput;
    }

	std::shared_ptr<Lumen::ILumenVolume> WaveFrontRenderer::CreateVolume(const std::string& a_FilePath)
	{
		//TODO tell optix to create a volume acceleration structure.
		auto volume = std::make_shared<PTVolume>(a_FilePath, m_ServiceLocator);

		uint32_t geomFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE };

		OptixAccelBuildOptions buildOptions = {};
		buildOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
		buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
		buildOptions.motionOptions = {};

		OptixAabb aabb = { -1.5f, -1.5f, -1.5f, 1.5f, 1.5f, 1.5f };

		auto grid = volume->GetHandle()->grid<float>();
		auto bbox = grid->worldBBox();

		nanovdb::Vec3<double> temp = bbox.min();
		float bboxMinX = bbox.min()[0];
		float bboxMinY = bbox.min()[1];
		float bboxMinZ = bbox.min()[2];
		float bboxMaxX = bbox.max()[0];
		float bboxMaxY = bbox.max()[1];
		float bboxMaxZ = bbox.max()[2];

		aabb = { bboxMinX, bboxMinY, bboxMinZ, bboxMaxX, bboxMaxY, bboxMaxZ };

		MemoryBuffer aabb_buffer(sizeof(OptixAabb));
		aabb_buffer.Write(aabb);

		OptixBuildInput buildInput = {};
		buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
		buildInput.customPrimitiveArray.aabbBuffers = &*aabb_buffer;
		buildInput.customPrimitiveArray.numPrimitives = 1;
		buildInput.customPrimitiveArray.flags = geomFlags;
		buildInput.customPrimitiveArray.numSbtRecords = 1;

		volume->m_AccelerationStructure = m_OptixSystem->BuildGeometryAccelerationStructure(buildOptions, buildInput);

		//This has been moved to volume instance
		//volume->m_SceneDataTableEntry = m_Table->AddEntry<DeviceVolume>();
		//auto& entry = volume->m_SceneDataTableEntry.GetData();
		//entry.m_Grid = grid;

		return volume;
	}

    void WaveFrontRenderer::TraceFrame()
    {
        static Timer wavefrontTimer;
    	//Track frame time.
        Timer timer;    	
        CHECKLASTCUDAERROR;

        //Retrieve the acceleration structure and scene data table once.
        m_OptixSystem->UpdateSBT();
        CHECKLASTCUDAERROR;
        auto* sceneDataTableAccessor = std::static_pointer_cast<PTScene>(m_Scene)->GetSceneDataTableAccessor();
    	
        CHECKLASTCUDAERROR;

        auto accelerationStructure = std::static_pointer_cast<PTScene>(m_Scene)->GetSceneAccelerationStructure();
        cudaDeviceSynchronize();
        CHECKLASTCUDAERROR;

        m_CurrentFrameStats.m_Times["Update SBT"] = timer.measure(TimeUnit::MICROS);
        timer.reset();

        const unsigned int numLightsInScene = m_LightDataBuffer->BuildLightDataBuffer(std::static_pointer_cast<PTScene>(m_Scene), sceneDataTableAccessor);
    	
        //Don't render if there is no light in the scene as everything will be black anyway.
        if (numLightsInScene == 0)
        {
            // Are you sure there are lights in the scene? Then restart the application, weird bugs man.
            __debugbreak();
            return;
        }

        m_CurrentFrameStats.m_Times["Light Uploading"] = timer.measure(TimeUnit::MICROS);
        timer.reset();
    	
        bool recordingSnapshot = m_StartSnapshot;
        if (m_StartSnapshot)
        {
            // Replacing the snapshot with a non-null one will start recording requested features.
            m_FrameSnapshot = std::make_unique<FrameSnapshot>();
            m_StartSnapshot = false;
        }

        m_CurrentFrameStats.m_Times["Snapshot"] = timer.measure(TimeUnit::MICROS);
        timer.reset();

        bool resizeBuffers = false, resizeOutputBuffer = false;
        {
            CHECKLASTCUDAERROR;

            // Lock the settings mutex while we copy its data
            std::lock_guard lock(m_SettingsUpdateMutex);

            // Also check if the render resolution or output resolution have changed,
            // since those would require resizing buffers in this frame

            // TODO: Render and output resolution comparisons
            if (m_Settings.renderResolution.x != m_IntermediateSettings.renderResolution.x ||
                m_Settings.renderResolution.y != m_IntermediateSettings.renderResolution.y)
            {
                resizeBuffers = true;
                m_IntermediateSettings.outputResolution = m_IntermediateSettings.renderResolution;
            }

            if (m_Settings.outputResolution.x != m_IntermediateSettings.outputResolution.x ||
                m_Settings.outputResolution.y != m_IntermediateSettings.outputResolution.y)
            {
                resizeOutputBuffer = true;
            }

            m_Settings = m_IntermediateSettings;
        }

        if (resizeBuffers)
        {
            m_DeferredOpenGLCalls.push([this]() {ResizeBuffers(); });
        }

        if (resizeOutputBuffer)
        {
            // TODO: Resizing openGL output buffer using output resolution
            //Set up the OpenGL output buffer.
            m_DeferredOpenGLCalls.push([this]()
                {
                    m_OutputBuffer->Resize(m_IntermediateSettings.outputResolution.x, m_IntermediateSettings.outputResolution.y);
                });
        }
        CHECKLASTCUDAERROR;

        m_CurrentFrameStats.m_Times["Resize Buffers"] = timer.measure(TimeUnit::MICROS);
        timer.reset();

        //Index of the current and last frame to access buffers.
        const auto currentIndex = m_FrameIndex;
        const auto temporalIndex = m_FrameIndex == 1 ? 0 : 1;

        // TODO: Set number of pixels using render resolution
        //Data needed in the algorithms.
        const uint32_t numPixels = m_Settings.renderResolution.x * m_Settings.renderResolution.y;
        CHECKLASTCUDAERROR;

        //TODO: Is this the best spot to stall the rendering thread to update resources? I've no clue.
        WaitForDeferredCalls();
        CHECKLASTCUDAERROR;

        m_CurrentFrameStats.m_Times["Deferred Calls 1"] = timer.measure(TimeUnit::MICROS);
        timer.reset();



        //Prepare separate pixel buffer(s) for CUDA operations.
        //Get the surface objects to use for the different light channels into a single variable.
        std::array<cudaSurfaceObject_t, s_numLightChannels> pixelBuffers{};
        for (unsigned int lightChannelIndex = 0; lightChannelIndex < s_numLightChannels; ++lightChannelIndex)
        {
            m_PixelBufferSeparate[lightChannelIndex]->Map(lightChannelIndex);
            pixelBuffers[lightChannelIndex] = m_PixelBufferSeparate[lightChannelIndex]->GetSurfaceObject();
        }

        //Prepare combined pixel buffer for CUDA operations.
        m_PixelBufferCombined->Map();

        //Start by clearing the data from the previous frame.
        for (auto& buffer : m_PixelBufferSeparate) { buffer->Clear(); }

        //Only clean the merged buffer if no blending is enabled.
        if (!m_Settings.blendOutput) { m_PixelBufferCombined->Clear(); }

        m_DepthBuffer->Map();
        m_JitterBuffer->Map();
        m_MotionVectorBuffer->Map();
        m_NormalRoughnessBuffer->Map();

        cudaDeviceSynchronize();
        CHECKLASTCUDAERROR;

        m_CurrentFrameStats.m_Times["Map + Clear Buffers"] = timer.measure(TimeUnit::MICROS);
        timer.reset();

        //Generate camera rays.
        glm::vec3 eye, u, v, w;
        // TODO: Aspect ratio set with render resolution
        m_Scene->m_Camera->SetAspectRatio(static_cast<float>(m_Settings.renderResolution.x) / static_cast<float>(m_Settings.renderResolution.y));
        m_Scene->m_Camera->GetVectorData(eye, u, v, w);

        //Camera forward direction.
        const float3 camForward = { w.x, w.y, w.z };
        const float3 camPosition = { eye.x, eye.y, eye.z };

        float3 eyeCuda, uCuda, vCuda, wCuda;
        eyeCuda = make_float3(eye.x, eye.y, eye.z);
        uCuda = make_float3(u.x, u.y, u.z);
        vCuda = make_float3(v.x, v.y, v.z);
        wCuda = make_float3(w.x, w.y, w.z);

        //printf("Camera pos: %f %f %f\n", camPosition.x, camPosition.y, camPosition.z);

        //Increment framecount each frame.
        static unsigned frameCount = 0;
        ++frameCount;

        const WaveFront::PrimRayGenLaunchParameters::DeviceCameraData cameraData(eyeCuda, uCuda, vCuda, wCuda);
        auto rayPtr = m_Rays.GetDevicePtr<AtomicBuffer<IntersectionRayData>>();
        const PrimRayGenLaunchParameters primaryRayGenParams(
            m_Settings.renderResolution,        // TODO: Raygen executed with render resolution
            cameraData,
            rayPtr, frameCount);
        //GeneratePrimaryRays(primaryRayGenParams, m_PixelBufferCombined->GetSurfaceObject());
        m_JitterBuffer->Map();
        GeneratePrimaryRays(primaryRayGenParams, m_JitterBuffer->GetSurfaceObject());
        cudaDeviceSynchronize();    //explode
        CHECKLASTCUDAERROR;

        //Set the atomic counter for primary rays to the amount of pixels.
        SetAtomicCounter<IntersectionRayData>(&m_Rays, numPixels);

        m_CurrentFrameStats.m_Times["Primary Ray Generation"] = timer.measure(TimeUnit::MICROS);
        timer.reset();

        // TODO: render resolution used in snapshot->AddBuffer
        m_FrameSnapshot->AddBuffer([&]()
            {
                auto optixDenoiserInput = &(m_OptixDenoiser->ColorInput);
                auto OptixDenoiserAlbedoInput = &(m_OptixDenoiser->AlbedoInput);
                auto OptixDenoiserNormalInput = &(m_OptixDenoiser->NormalInput);
                auto optixDenoiszerOutput = &(m_OptixDenoiser->GetColorOutput());

                std::map<std::string, FrameSnapshot::ImageBuffer> resBuffers;
                m_DeferredOpenGLCalls.push([&]() {
                    resBuffers["Optix denoiser color input"].m_Memory = std::make_unique<CudaGLTexture>(GL_RGB32F, m_Settings.renderResolution.x,
                        m_Settings.renderResolution.y, 3 * sizeof(float));

                    resBuffers["Optix denoiser albedo input"].m_Memory = std::make_unique<CudaGLTexture>(GL_RGB32F, m_Settings.renderResolution.x,
                        m_Settings.renderResolution.y, 3 * sizeof(float));

                    resBuffers["Optix denoiser normal input"].m_Memory = std::make_unique<CudaGLTexture>(GL_RGB32F, m_Settings.renderResolution.x,
                        m_Settings.renderResolution.y, 3 * sizeof(float));

                    resBuffers["Optix denoiser color output"].m_Memory = std::make_unique<CudaGLTexture>(GL_RGB32F, m_Settings.renderResolution.x,
                        m_Settings.renderResolution.y, 3 * sizeof(float));
                    });
                WaitForDeferredCalls();

                SeparateOptixDenoiserBufferCPU(m_Settings.renderResolution.x * m_Settings.renderResolution.y,
                    optixDenoiserInput->GetDevicePtr<float3>(),
                    OptixDenoiserAlbedoInput->GetDevicePtr<float3>(),
                    OptixDenoiserNormalInput->GetDevicePtr<float3>(),
                    optixDenoiszerOutput->GetDevicePtr<float3>(),
                    resBuffers.at("Optix denoiser color input").m_Memory->GetDevicePtr<float3>(),
                    resBuffers.at("Optix denoiser albedo input").m_Memory->GetDevicePtr<float3>(),
                    resBuffers.at("Optix denoiser normal input").m_Memory->GetDevicePtr<float3>(),
                    resBuffers.at("Optix denoiser color output").m_Memory->GetDevicePtr<float3>()
                );

                return resBuffers;
            });

        //Clear the surface data that contains information from the second last frame so that it can be reused by this frame.
        cudaMemset(m_SurfaceData[currentIndex].GetDevicePtr(), 0, sizeof(SurfaceData) * numPixels);
        cudaDeviceSynchronize();
        CHECKLASTCUDAERROR;

        m_CurrentFrameStats.m_Times["Snapshot + Clear SurfaceData"] = timer.measure(TimeUnit::MICROS);
        timer.reset();

        //Set the counters back to 0 for intersections and shadow rays.
        const unsigned counterDefault = 0;
        SetAtomicCounter<ShadowRayData>(&m_ShadowRays, counterDefault);
		SetAtomicCounter<ShadowRayData>(&m_VolumetricShadowRays, counterDefault);
        //SetAtomicCounter<IntersectionData>(&m_IntersectionData, counterDefault);  //No need to reset as this buffer is not used as an atomic buffer for now.
		SetAtomicCounter<VolumetricIntersectionData>(&m_VolumetricIntersectionData, counterDefault);
        CHECKLASTCUDAERROR;



        //Pass the buffers to the optix shader for shading.
        OptixLaunchParameters rayLaunchParameters {};

        rayLaunchParameters.m_TraversableHandle = accelerationStructure;
        rayLaunchParameters.m_ResolutionAndDepth = uint3{ m_Settings.renderResolution.x, m_Settings.renderResolution.y, m_Settings.depth };
        rayLaunchParameters.m_IntersectionRayBatch = m_Rays.GetDevicePtr<AtomicBuffer<IntersectionRayData>>();
        rayLaunchParameters.m_IntersectionBuffer = m_IntersectionData.GetDevicePtr<AtomicBuffer<IntersectionData>>();
        rayLaunchParameters.m_VolumetricIntersectionBuffer = m_VolumetricIntersectionData.GetDevicePtr<AtomicBuffer<VolumetricIntersectionData>>();
        rayLaunchParameters.m_SceneData = sceneDataTableAccessor;
        rayLaunchParameters.m_MinMaxDistance = { 0.01f, 5000.f };
        rayLaunchParameters.m_TraceType = RayType::INTERSECTION_RAY;
        // TODO: render resolution used in ray launch params

        //Set the amount of rays to trace. Initially same as screen size.
        auto numIntersectionRays = numPixels;

        auto seed = WangHash(frameCount);

        const glm::vec2 renderDistanceMinMax = m_Scene->m_Camera->GetMinMaxRenderDistance();

        float2 minMaxDepth = make_float2(renderDistanceMinMax.x, renderDistanceMinMax.y);

        m_CurrentFrameStats.m_Times["Reset Counters"] = timer.measure(TimeUnit::MICROS);
        timer.reset();

        /*
         * Resolve rays and shade at every depth.
         */
        for (unsigned depth = 0; depth < m_Settings.depth && numIntersectionRays > 0; ++depth)
        {
            //Set the atomic intersection buffer to also have the right counter.
            SetAtomicCounter<IntersectionData>(&m_IntersectionData, numIntersectionRays);
        	
            //Tell Optix to resolve the primary rays that have been generated.
            m_OptixSystem->TraceRays(numIntersectionRays, rayLaunchParameters);
            cudaDeviceSynchronize();
            CHECKLASTCUDAERROR;

            /*
             * Calculate the surface data for this depth.
             */
            //unsigned numIntersections = 0;        //Note; Not currently used as atomic buffer. One intersection per ray.
            //numIntersections = GetAtomicCounter<IntersectionData>(&m_IntersectionData);

            //1 and 2 are used for the first intersection and remembered for temporal use.
            const auto surfaceDataBufferIndex = (depth == 0 ? currentIndex : 2);   

        	//if(numIntersections > 0)  //Note: This is always true because the loop already does it.
        	//{

                //pass depth buffer into extract surface data
                ExtractSurfaceData(
                    numIntersectionRays,    //Note: rays is always equal to num intersections. Even missed rays return an intersection that is empty.
                    m_IntersectionData.GetDevicePtr<AtomicBuffer<IntersectionData>>(),
                    m_Rays.GetDevicePtr<AtomicBuffer<IntersectionRayData>>(),
                    m_SurfaceData[surfaceDataBufferIndex].GetDevicePtr<SurfaceData>(),
                    m_DepthBuffer->GetSurfaceObject(),
                    //m_PixelBufferCombined->GetSurfaceObject(), //To debug the depth buffer.
                    m_Settings.renderResolution,
                    sceneDataTableAccessor,
                    minMaxDepth,
                    depth);

                //ExtractSurfaceData(extractionParams);

                cudaDeviceSynchronize();
                CHECKLASTCUDAERROR;

        	//}
        	
            unsigned numVolumeIntersections = 0;
            m_VolumetricIntersectionData.Read(&numVolumeIntersections, sizeof(numVolumeIntersections), 0);

            const auto volumetricDataBufferIndex = 0;

        	//Ensure that there is actually volumes to extract data from.
        	if(numVolumeIntersections > 0)
        	{
                ExtractVolumetricData(
                    numVolumeIntersections,
                    m_VolumetricIntersectionData.GetDevicePtr<AtomicBuffer<VolumetricIntersectionData>>(),
                    m_Rays.GetDevicePtr<AtomicBuffer<IntersectionRayData>>(),
                    m_VolumetricData[volumetricDataBufferIndex].GetDevicePtr<VolumetricData>(),
                    m_Settings.renderResolution,
                    sceneDataTableAccessor);

                cudaDeviceSynchronize();
                CHECKLASTCUDAERROR;
        	}

            //motion vector generation
            if (depth == 0)
            {

                glm::mat4 previousFrameMatrix, currentFrameMatrix;
                m_Scene->m_Camera->GetMatrixData(previousFrameMatrix, currentFrameMatrix);
                sutil::Matrix4x4 prevFrameMatrixArg = ConvertGLMtoSutilMat4(previousFrameMatrix);

                glm::mat4 projectionMatrix = m_Scene->m_Camera->GetProjectionMatrix();
                sutil::Matrix4x4 projectionMatrixArg = ConvertGLMtoSutilMat4(projectionMatrix);

                MotionVectorsGenerationData motionVectorsGenerationData;
                motionVectorsGenerationData.m_MotionVectorBuffer = m_MotionVectorBuffer->GetSurfaceObject();
                motionVectorsGenerationData.m_CurrentSurfaceData = m_SurfaceData[currentIndex].GetDevicePtr<SurfaceData>();
                // TODO: Render resolution used in motion vector generation data
                motionVectorsGenerationData.m_RenderResolution = make_uint2(m_Settings.renderResolution.x, m_Settings.renderResolution.y);
                motionVectorsGenerationData.m_PrevViewMatrix = prevFrameMatrixArg.inverse();
                motionVectorsGenerationData.m_ProjectionMatrix = projectionMatrixArg;

                GenerateMotionVectors(motionVectorsGenerationData);
                CHECKLASTCUDAERROR

            }

            /*
             * Call the shading kernels.
             */

            ShadingLaunchParameters shadingLaunchParams(
                uint3{ m_Settings.renderResolution.x, m_Settings.renderResolution.y, m_Settings.depth },
                m_SurfaceData[surfaceDataBufferIndex].GetDevicePtr<SurfaceData>(),
                m_SurfaceData[temporalIndex].GetDevicePtr<SurfaceData>(),
                m_VolumetricData[volumetricDataBufferIndex].GetDevicePtr<VolumetricData>(),
                m_MotionVectorBuffer->GetSurfaceObject(),
                m_LightDataBuffer->GetDataBuffer(),
                accelerationStructure,
                m_OptixSystem.get(),
                depth,
                seed,
                m_Rays.GetDevicePtr<AtomicBuffer<IntersectionRayData>>(),
                m_ShadowRays.GetDevicePtr<AtomicBuffer<ShadowRayData>>(),
                m_VolumetricShadowRays.GetDevicePtr<AtomicBuffer<ShadowRayData>>(),
                m_ReSTIR.get(),
                pixelBuffers);

            shadingLaunchParams.m_FrameStats = &m_CurrentFrameStats;
            //Reset the ray buffer so that indirect shading can fill it again.
            ResetAtomicBuffer<IntersectionRayData>(&m_Rays);
            cudaDeviceSynchronize();
            CHECKLASTCUDAERROR;
        	
            Shade(shadingLaunchParams);
            cudaDeviceSynchronize();
            CHECKLASTCUDAERROR;

            //Set the number of intersection rays to the size of the ray buffer.
            numIntersectionRays = GetAtomicCounter<IntersectionRayData>(&m_Rays);

            //Clear the surface data at depth 2 (the one that is overwritten each wave).
            cudaMemset(m_SurfaceData[2].GetDevicePtr(), 0, sizeof(SurfaceData) * numPixels);
            cudaDeviceSynchronize();
            CHECKLASTCUDAERROR;

            //Reset the intersection data so that the next frame can re-fill it.
            ResetAtomicBuffer<IntersectionData>(&m_IntersectionData);
			ResetAtomicBuffer<VolumetricIntersectionData>(&m_VolumetricIntersectionData);

            //Swap the ReSTIR buffers around.
            m_ReSTIR->SwapBuffers();

            //Switch up the seed.
            seed = WangHash(seed);
        }

        m_CurrentFrameStats.m_Times["Wavefront Iteration"] = timer.measure(TimeUnit::MICROS);
        timer.reset();

        //The amount of shadow rays to trace.
        unsigned numShadowRays = GetAtomicCounter<ShadowRayData>(&m_ShadowRays);

        if (numShadowRays > 0)
        {

            //The settings for shadow ray resolving.
            OptixLaunchParameters shadowRayLaunchParameters = rayLaunchParameters; //Copy settings from the intersection rays.
            shadowRayLaunchParameters.m_ShadowRayBatch = m_ShadowRays.GetDevicePtr<AtomicBuffer<ShadowRayData>>();
            shadowRayLaunchParameters.m_OutputChannels = pixelBuffers;
            shadowRayLaunchParameters.m_TraceType = RayType::SHADOW_RAY;

            //Tell optix to resolve the shadow rays.
            m_OptixSystem->TraceRays(numShadowRays, shadowRayLaunchParameters);
            cudaDeviceSynchronize();
            CHECKLASTCUDAERROR;
        }

        m_CurrentFrameStats.m_Times["Shadow Rays"] = timer.measure(TimeUnit::MICROS);
        timer.reset();

		//The amount of shadow rays to trace.
		unsigned numVolumetricShadowRays = GetAtomicCounter<ShadowRayData>(&m_VolumetricShadowRays);

		if (numVolumetricShadowRays > 0)
		{
			//The settings for shadow ray resolving.
			OptixLaunchParameters shadowRayLaunchParameters = rayLaunchParameters;
			shadowRayLaunchParameters.m_ShadowRayBatch = m_VolumetricShadowRays.GetDevicePtr<AtomicBuffer<ShadowRayData>>();
            shadowRayLaunchParameters.m_OutputChannels = pixelBuffers;
			shadowRayLaunchParameters.m_TraceType = RayType::SHADOW_RAY;

			m_OptixSystem->TraceRays(numVolumetricShadowRays, shadowRayLaunchParameters);
			cudaDeviceSynchronize();
			CHECKLASTCUDAERROR;
		}

        m_CurrentFrameStats.m_Times["Volume Shadow Rays"] = timer.measure(TimeUnit::MICROS);
        timer.reset();
    	
        //Post-processing
        {
            m_PixelBufferUpscaled->Map();
            // TODO: render and output resolution used in post processing
            PostProcessLaunchParameters postProcessLaunchParams(
                m_Settings.renderResolution,
                m_Settings.outputResolution,
                pixelBuffers,
                m_PixelBufferCombined->GetSurfaceObject(),
                m_IntermediateOutputBuffer.GetDevicePtr<uchar4>(),  //might be uneccesary
                m_Settings.blendOutput,
                m_BlendCounter
            );
            //Post processing using CUDA kernel.
            //PostProcess(postProcessLaunchParams);

            //Unmap other buffers for use in DLSS
            /*m_PixelBufferSeparate[0]->Unmap();
            m_PixelBufferSeparate[1]->Unmap();
            m_PixelBufferCombined->Unmap();
            m_DepthBuffer->Unmap();
            m_JitterBuffer->Unmap();
            m_MotionVectorBuffer->Unmap();
            m_NormalRoughnessBuffer->Unmap();*/

            if (/*m_DenoiserSettings.m_UseNRD*/true)
            {
                NRDWrapperEvaluateParams nrdParams = {};
                nrdParams.m_Camera = m_Scene->m_Camera.get();
                m_NRD->Denoise(nrdParams);
            }

            if (m_DenoiserSettings.m_UseOptix)
            {
                postProcessLaunchParams.m_BlendOutput = false;
            }

            MergeOutput(postProcessLaunchParams);
            cudaDeviceSynchronize();
            CHECKLASTCUDAERROR;

            if (m_DenoiserSettings.m_UseOptix)
            {
                OptixDenoiserLaunchParameters optixDenoiserLaunchParams(
                    m_Settings.renderResolution,
                    m_SurfaceData[currentIndex].GetDevicePtr<SurfaceData>(),
                    m_PixelBufferCombined->GetSurfaceObject(),
                    m_OptixDenoiser->ColorInput.GetDevicePtr<float3>(),
                    m_OptixDenoiser->AlbedoInput.GetDevicePtr<float3>(),
                    m_OptixDenoiser->NormalInput.GetDevicePtr<float3>(),
                    m_OptixDenoiser->FlowInput.GetDevicePtr<float2>(),
                    m_OptixDenoiser->GetColorOutput().GetDevicePtr<float3>()
                );
                OptixDenoiserInitParams optixDenoiserInitParams;
                optixDenoiserInitParams.m_InputWidth = m_Settings.renderResolution.x;
                optixDenoiserInitParams.m_InputHeight = m_Settings.renderResolution.y;
                optixDenoiserInitParams.m_ServiceLocator = &m_ServiceLocator;
                optixDenoiserInitParams.m_UseAlbedo = m_DenoiserSettings.m_OptixAlbedo;
                optixDenoiserInitParams.m_UseNormal = m_DenoiserSettings.m_OptixNormal;
                optixDenoiserInitParams.m_UseTemporalData = m_DenoiserSettings.m_OptixTemporal;
                OptixDenoiserDenoiseParams optixDenoiserParams = {};
                optixDenoiserParams.m_InitParams = optixDenoiserInitParams;
                optixDenoiserParams.m_PostProcessLaunchParams = &postProcessLaunchParams;
                optixDenoiserParams.m_OptixDenoiserLaunchParams = &optixDenoiserLaunchParams;
                optixDenoiserParams.m_BlendOutput = m_Settings.blendOutput;
                optixDenoiserParams.m_BlendCount = m_BlendCounter;
                m_OptixDenoiser->Denoise(optixDenoiserParams);
            }


            //OptixDenoiserDenoiseParams optixDenoiserParams = {}; 
            //optixDenoiserParams.m_PostProcessLaunchParams = &postProcessLaunchParams; 
            //optixDenoiserParams.m_ColorInput = m_OptixDenoiser->ColorInput.GetCUDAPtr();
            //optixDenoiserParams.m_AlbedoInput = m_OptixDenoiser->AlbedoInput.GetCUDAPtr();
            //optixDenoiserParams.m_NormalInput = m_OptixDenoiser->NormalInput.GetCUDAPtr();
            //optixDenoiserParams.m_FlowInput = m_OptixDenoiser->FlowInput.GetCUDAPtr();
            //optixDenoiserParams.m_PrevColorOutput = m_OptixDenoiser->GetPrevColorOutput().GetCUDAPtr();
            //optixDenoiserParams.m_ColorOutput = m_OptixDenoiser->GetColorOutput().GetCUDAPtr();
            //m_OptixDenoiser->Denoise(optixDenoiserParams); 
            //CHECKLASTCUDAERROR;

            //FinishOptixDenoising(optixDenoiserLaunchParams);
            CHECKLASTCUDAERROR;

            //Trim down post processing params
            //Replace use of post process params in WriteToOutput with OutputWrite params

            //Cuda should no longer be operating on the pixel-buffers.
            //Unmap separate channel pixel-buffer
            for (auto& buffer : m_PixelBufferSeparate) { buffer->Unmap(); }

            //Unmap other buffers for use in DLSS
            m_PixelBufferCombined->Unmap();
            m_DepthBuffer->Unmap();
            m_JitterBuffer->Unmap();
            m_MotionVectorBuffer->Unmap();
            m_NormalRoughnessBuffer->Unmap();

             //TODO: Render and output resolutions used in DLSS init params
            if (m_DLSS)    // Should really do this differently, DLSS not initialized properly yet
            {
                std::shared_ptr<DLSSWrapperInitParams> params = m_DLSS->GetDLSSParams();
                params->m_InputImageWidth = m_Settings.renderResolution.x;
                params->m_InputImageHeight = m_Settings.renderResolution.y;
                params->m_OutputImageWidth = m_Settings.outputResolution.x;
                params->m_OutputImageHeight = m_Settings.outputResolution.y;
                params->m_DLSSMode = static_cast<DLSSMode>(m_DlssMode);

                CHECKLASTCUDAERROR;

                m_PixelBufferUpscaled->Unmap();
                m_PixelBufferCombined->Unmap();
                m_DepthBuffer->Unmap();
                m_MotionVectorBuffer->Unmap();
                m_JitterBuffer->Unmap();

                m_DLSS->EvaluateDLSS(m_D3D11PixelBufferUpscaled, m_D3D11PixelBufferCombined, m_D3D11DepthBuffer, m_D3D11MotionVectorBuffer, m_D3D11JitterBuffer);
            }


            m_PixelBufferCombined->Map();
            //m_PixelBufferUpscaled->Map();
            WriteOutputParams writeParams(
                m_Settings.outputResolution,
                m_PixelBufferCombined->GetSurfaceObject(),
                //m_PixelBufferUpscaled->GetSurfaceObject(),
                m_IntermediateOutputBuffer.GetDevicePtr<uchar4>()
            );

            //TODO: move to after DLSS and NRD runs... (Requires mapping to use in CUDA again).
            WriteToOutput(writeParams);     //Breaks here when render and output resolution dont match. Uses output resolution
            cudaDeviceSynchronize();
            CHECKLASTCUDAERROR;

        }

        m_CurrentFrameStats.m_Times["Post Processing"] = timer.measure(TimeUnit::MICROS);
        timer.reset();

        // Critical scope for updating the output texture
        {
            std::unique_lock guard(m_OutputBufferMutex); // Take ownership of the mutex, locking it

            CHECKLASTCUDAERROR;

            // Perform a GPU to GPU copy, from the intermediate output buffer to the real output buffer
            CHECKCUDAERROR(cudaMemcpy(m_OutputBuffer->GetDevicePtr<void>(), m_IntermediateOutputBuffer.GetDevicePtr(),
                m_IntermediateOutputBuffer.GetSize(), cudaMemcpyKind::cudaMemcpyDeviceToDevice));

            CHECKCUDAERROR(cudaDeviceSynchronize());

            // Once the memcpy is complete, the lock guard releases the mutex
        }
        CHECKLASTCUDAERROR;

        m_CurrentFrameStats.m_Times["Update Output Texture"] = timer.measure(TimeUnit::MICROS);
        timer.reset();

        // The display thread might wait on the memcpy that was just performed.
        // We bump the thread by calling notify all on the same condition_variable it uses
        m_OutputCondition.notify_all(); 

        //If blending is enabled, increment blend counter.
        if (m_Settings.blendOutput)
        {
            ++m_BlendCounter;
        }

        //Change frame index 0..1
        ++m_FrameIndex;
        if (m_FrameIndex == 2)
        {
            m_FrameIndex = 0;
        }

        m_Scene->m_Camera->UpdatePreviousFrameMatrix();
        ++frameCount;

        //m_OptixDenoiser->UpdateDebugTextures();
        //m_DeferredOpenGLCalls.push([&]() {
        //    //m_DebugTexture = m_OptixDenoiser->m_OptixDenoiserInputTex.m_Memory->GetTexture();
        //    //m_DebugTexture = m_OptixDenoiser->m_OptixDenoiserAlbedoInputTex.m_Memory->GetTexture();
        //    //m_DebugTexture = m_OptixDenoiser->m_OptixDenoiserNormalInputTex.m_Memory->GetTexture();
        //    m_DebugTexture = m_OptixDenoiser->m_OptixDenoiserOutputTex.m_Memory->GetTexture();
        //    });
        //WaitForDeferredCalls();

        m_CurrentFrameStats.m_Times["Finalize"] = timer.measure(TimeUnit::MICROS);
        timer.reset();

        // TODO: Weird debug code. Yeet?
        //m_DebugTexture = m_OutputBuffer->GetTexture();
        //#if defined(_DEBUG)
        //m_MotionVectors.GenerateDebugTextures();
        ////m_DebugTexture = m_MotionVectors.GetMotionVectorDirectionsTex();
        //m_DebugTexture = m_MotionVectors.GetMotionVectorMagnitudeTex();

        /*m_OptixDenoiser->UpdateDebugTextures();
        m_DebugTexture = m_OptixDenoiser->m_OptixDenoiserInputTex.m_Memory->GetTexture();*/

        m_SnapshotReady = recordingSnapshot;

        m_CurrentFrameStats.m_Times["Total Frame Time"] = wavefrontTimer.measure(TimeUnit::MICROS);
        wavefrontTimer.reset();
    	
        FinalizeFrameStats();

        CHECKLASTCUDAERROR;

        //tonemapping should be done before DLSS
        //DLSS performs better on LDR data as opposed to HDR

        
    }

    std::unique_ptr<MemoryBuffer> WaveFrontRenderer::InterleaveVertexData(const PrimitiveData& a_MeshData) const
    {
        std::vector<Vertex> vertices;

        for (size_t i = 0; i < a_MeshData.m_Positions.Size(); i++)
        {
            auto& v = vertices.emplace_back();
            v.m_Position = make_float3(a_MeshData.m_Positions[i].x, a_MeshData.m_Positions[i].y, a_MeshData.m_Positions[i].z);
            if (!a_MeshData.m_TexCoords.Empty())
                v.m_UVCoord = make_float2(a_MeshData.m_TexCoords[i].x, a_MeshData.m_TexCoords[i].y);
            if (!a_MeshData.m_Normals.Empty())
                v.m_Normal = make_float3(a_MeshData.m_Normals[i].x, a_MeshData.m_Normals[i].y, a_MeshData.m_Normals[i].z);
            if (!a_MeshData.m_Tangents.Empty())
                v.m_Tangent = make_float4(a_MeshData.m_Tangents[i].x, a_MeshData.m_Tangents[i].y, a_MeshData.m_Tangents[i].z, a_MeshData.m_Tangents[i].w);
        }
        return std::make_unique<MemoryBuffer>(vertices);
    }

    void WaveFrontRenderer::StartRendering()
    {
        m_PathTracingThread = std::thread([&]()
            {
                while (!m_StopRendering)
                    TraceFrame();

            });       
    }

    void WaveFrontRenderer::PerformDeferredOperations()
    {
        CHECKLASTCUDAERROR;

        m_OutputBuffer->Map(); // Honestly, curse OpenGL
        CHECKLASTCUDAERROR;

        while (!m_DeferredOpenGLCalls.empty())
        {
            m_DeferredOpenGLCalls.front()();
            m_DeferredOpenGLCalls.pop();
        }

        m_OGLCallCondition.notify_all();
    }

    Lumen::SceneManager::GLTFResource WaveFrontRenderer::OpenCustomFileFormat(const std::string& a_OriginalFilePath)
    {
        std::filesystem::path p(a_OriginalFilePath);
        p.replace_extension(LumenPTModelConverter::ms_ExtensionName);

        return m_ModelConverter.LoadFile(p.string());
    }

    Lumen::SceneManager::GLTFResource WaveFrontRenderer::CreateCustomFileFormat(const std::string& a_OriginalFilePath)
    {
        return m_ModelConverter.ConvertGLTF(a_OriginalFilePath);
    }

    std::unique_ptr<Lumen::ILumenPrimitive> WaveFrontRenderer::CreatePrimitive(PrimitiveData& a_PrimitiveData)
    {
        //TODO let optix build the acceleration structure and return the handle.
        std::unique_ptr<MemoryBuffer> vertexBuffer;
        if (!a_PrimitiveData.m_Interleaved)
            vertexBuffer = InterleaveVertexData(a_PrimitiveData);
        else
            vertexBuffer = std::make_unique<MemoryBuffer>(a_PrimitiveData.m_VertexBinary);

        cudaDeviceSynchronize();
        auto err = cudaGetLastError();
        std::vector<uint32_t> correctedIndices;

        if (a_PrimitiveData.m_IndexSize != 4)
        {
            VectorView<uint16_t, uint8_t> indexView(a_PrimitiveData.m_IndexBinary);

            for (size_t i = 0; i < indexView.Size(); i++)
            {
                correctedIndices.push_back(indexView[i]);
            }
        }
        //Gotta copy over even when they are 32 bit.
        else
        {
            VectorView<uint32_t, uint8_t> indexView(a_PrimitiveData.m_IndexBinary);
            for (size_t i = 0; i < indexView.Size(); i++)
            {
                correctedIndices.push_back(indexView[i]);
            }
        }
        
        //printf("Index buffer Size %i \n", static_cast<int>(correctedIndices.size()));
        std::unique_ptr<MemoryBuffer> indexBuffer = std::make_unique<MemoryBuffer>(correctedIndices);

        uint32_t numIndices = correctedIndices.size();
        const size_t memSize = (numIndices / 3) * sizeof(bool);
        std::unique_ptr<MemoryBuffer> emissiveBuffer = std::make_unique<MemoryBuffer>(memSize); //might be wrong

        //Initialize with false so that nothing is emissive by default.
        cudaMemset(emissiveBuffer->GetDevicePtr(), 0, memSize);

        unsigned int numLights = 0; //number of emissive triangles in this primitive

        auto emissiveColor = std::static_pointer_cast<PTMaterial>(a_PrimitiveData.m_Material)->GetEmissiveColor();
        if (emissiveColor != glm::vec3(0.f))
        {

            cudaDeviceSynchronize();
            CHECKLASTCUDAERROR;
        	
            FindEmissives(
                vertexBuffer->GetDevicePtr<Vertex>(),
                indexBuffer->GetDevicePtr<uint32_t>(),
                emissiveBuffer->GetDevicePtr<bool>(),
                std::static_pointer_cast<PTMaterial>(a_PrimitiveData.m_Material)->GetDeviceMaterial(),
                numIndices,
                numLights
            );

            cudaDeviceSynchronize();
            CHECKLASTCUDAERROR;
        }
        


        unsigned int geomFlags = OPTIX_GEOMETRY_FLAG_NONE;

        OptixAccelBuildOptions buildOptions = {};
        buildOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        buildOptions.motionOptions = {};

        OptixBuildInput buildInput = {};
        buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        buildInput.triangleArray.indexBuffer = **indexBuffer;
        buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        buildInput.triangleArray.indexStrideInBytes = 0;
        buildInput.triangleArray.numIndexTriplets = correctedIndices.size() / 3;
        buildInput.triangleArray.vertexBuffers = &**vertexBuffer;
        buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        buildInput.triangleArray.vertexStrideInBytes = sizeof(Vertex);
        buildInput.triangleArray.numVertices = vertexBuffer->GetSize() / sizeof(Vertex);
        buildInput.triangleArray.numSbtRecords = 1;
        buildInput.triangleArray.flags = &geomFlags;

        auto gAccel = m_OptixSystem->BuildGeometryAccelerationStructure(buildOptions, buildInput);

        std::unique_ptr<PTPrimitive> prim = std::make_unique<PTPrimitive>(
            std::move(vertexBuffer), 
            std::move(indexBuffer), 
            std::move(emissiveBuffer), 
            std::move(gAccel));
        prim->m_Material = a_PrimitiveData.m_Material;
        prim->m_ContainEmissive = numLights > 0 ? true : false;
        prim->m_NumLights = numLights;

        prim->m_DevicePrimitive.m_VertexBuffer = prim->m_VertBuffer->GetDevicePtr<Vertex>();
        prim->m_DevicePrimitive.m_IndexBuffer = prim->m_IndexBuffer->GetDevicePtr<unsigned int>();
        prim->m_DevicePrimitive.m_Material = std::static_pointer_cast<PTMaterial>(prim->m_Material)->GetDeviceMaterial();
        prim->m_DevicePrimitive.m_EmissiveBuffer = prim->m_BoolBuffer->GetDevicePtr<bool>();
        CHECKLASTCUDAERROR;

        return prim;
    }

    std::shared_ptr<Lumen::ILumenMesh> WaveFrontRenderer::CreateMesh(
        std::vector<std::shared_ptr<Lumen::ILumenPrimitive>>& a_Primitives)
    {
        //TODO Let optix build the medium level acceleration structure and return the mesh handle for it.

        auto mesh = std::make_shared<PTMesh>(a_Primitives, m_ServiceLocator);
        return mesh;
    }

    std::shared_ptr<Lumen::ILumenTexture> WaveFrontRenderer::CreateTexture(void* a_PixelData,
        uint32_t a_Width, uint32_t a_Height, bool a_Normalize)
    {
        static cudaChannelFormatDesc formatDesc = cudaCreateChannelDesc<uchar4>();
        return std::make_shared<PTTexture>(a_PixelData, formatDesc, a_Width, a_Height, a_Normalize);
    }

    std::shared_ptr<Lumen::ILumenMaterial> WaveFrontRenderer::CreateMaterial(
        const MaterialData& a_MaterialData)
    {
    	//Make sure textures are not nullptr.
        assert(a_MaterialData.m_ClearCoatRoughnessTexture);
        assert(a_MaterialData.m_ClearCoatTexture);
        assert(a_MaterialData.m_DiffuseTexture);
        assert(a_MaterialData.m_EmissiveTexture);
        assert(a_MaterialData.m_MetallicRoughnessTexture);
        assert(a_MaterialData.m_TintTexture);
        assert(a_MaterialData.m_TransmissionTexture);
        assert(a_MaterialData.m_NormalMap);

    	//Roughness may never be 0.
        assert(a_MaterialData.m_RoughnessFactor > 0.f);
    	
        auto mat = std::make_shared<PTMaterial>();
        mat->SetDiffuseColor(a_MaterialData.m_DiffuseColor);
        mat->SetDiffuseTexture(a_MaterialData.m_DiffuseTexture);
        mat->SetEmission(a_MaterialData.m_EmissionVal);
        mat->SetEmissiveTexture(a_MaterialData.m_EmissiveTexture);
        mat->SetMetalRoughnessTexture(a_MaterialData.m_MetallicRoughnessTexture);
        mat->SetNormalTexture(a_MaterialData.m_NormalMap);

        //Disney
        mat->SetTransmissionTexture(a_MaterialData.m_TransmissionTexture);
        mat->SetClearCoatTexture(a_MaterialData.m_ClearCoatTexture);
        mat->SetClearCoatRoughnessTexture(a_MaterialData.m_ClearCoatRoughnessTexture);
        mat->SetTintTexture(a_MaterialData.m_TintTexture);

        mat->SetTransmissionFactor(a_MaterialData.m_TransmissionFactor);
        mat->SetClearCoatFactor(a_MaterialData.m_ClearCoatFactor);
        mat->SetClearCoatRoughnessFactor(a_MaterialData.m_ClearCoatRoughnessFactor);
        mat->SetIndexOfRefraction(a_MaterialData.m_IndexOfRefraction);
        mat->SetSpecularFactor(a_MaterialData.m_SpecularFactor);
        mat->SetSpecularTintFactor(a_MaterialData.m_SpecularTintFactor);
        mat->SetSubSurfaceFactor(a_MaterialData.m_SubSurfaceFactor);
        mat->SetLuminance(a_MaterialData.m_Luminance);
        mat->SetAnisotropic(a_MaterialData.m_Anisotropic);
        mat->SetSheenFactor(a_MaterialData.m_SheenFactor);
        mat->SetSheenTintFactor(a_MaterialData.m_SheenTintFactor);
        mat->SetTintFactor(a_MaterialData.m_TintFactor);
        mat->SetTransmittanceFactor(a_MaterialData.m_Transmittance);

        mat->SetRoughnessFactor(a_MaterialData.m_RoughnessFactor);
        mat->SetMetallicFactor(a_MaterialData.m_MetallicFactor);

        CHECKLASTCUDAERROR;

        return mat;
    }

    std::shared_ptr<Lumen::ILumenScene> WaveFrontRenderer::CreateScene(SceneData a_SceneData)
    {
        return std::make_shared<PTScene>(a_SceneData, m_ServiceLocator);
    }
    
    void WaveFrontRenderer::InitNGX()
    {

        // TODO: render and output resolutions used in DLSS init
        m_DLSS = std::make_unique<DlssWrapper>();
        DLSSWrapperInitParams dlssInitParams;
        dlssInitParams.m_InputImageWidth = m_Settings.renderResolution.x;
        dlssInitParams.m_InputImageHeight = m_Settings.renderResolution.y;
        dlssInitParams.m_OutputImageWidth = m_Settings.outputResolution.x;
        dlssInitParams.m_OutputImageHeight = m_Settings.outputResolution.y;
        dlssInitParams.m_DLSSMode = static_cast<DLSSMode>(m_DlssMode);   //Maybe add DLSS mode to m_Settings
        dlssInitParams.m_pServiceLocator = &m_ServiceLocator;
        if (!m_DLSS->InitializeNGX(dlssInitParams)) 
        {
            printf("DLSS could not be initialized!\n");
        }

        //auto recommendedSettings = m_DLSS->GetRecommendedSettings(Uint2_c(m_Settings.outputResolution.x, m_Settings.outputResolution.y), dlssInitParams.m_DLSSMode);
        //SetRenderResolution({ recommendedSettings->m_OptimalRenderSize.m_X, recommendedSettings->m_OptimalRenderSize.m_Y });
        //WaitForDeferredCalls();
        //std::cout << "Render resolution after NGX init: " << recommendedSettings->m_OptimalRenderSize.m_X << " " << recommendedSettings->m_OptimalRenderSize.m_Y << std::endl;
    }

    WaveFrontRenderer::WaveFrontRenderer() : m_BlendCounter(0)
        , m_FrameIndex(0)
        , m_CUDAContext(nullptr)
        , m_StopRendering(false)
        , m_StartSnapshot(false)
        , m_SnapshotReady(false)
    {

    }

    WaveFrontRenderer::~WaveFrontRenderer()
    {

        // Stop the path tracing thread and join it into the main thread
        m_StopRendering = true;
        assert(m_PathTracingThread.joinable() && "The wavefront renderer was never used for rendering, and its being destroyed.");
        m_PathTracingThread.join();

        // Explicitly destroy the scene before the scene data table to avoid
        // Dereferencing invalid memory addresses
        m_Scene.reset();
    }

    unsigned WaveFrontRenderer::GetOutputTexture()
    {        
        std::unique_lock<std::mutex> lock(m_OutputBufferMutex);
        return m_OutputBuffer->GetTexture();
    }

    std::vector<uint8_t> WaveFrontRenderer::GetOutputTexturePixels(uint32_t& a_Width, uint32_t& a_Height)
    {
        std::lock_guard<std::mutex> lock(m_OutputBufferMutex);
        auto devPtr = m_OutputBuffer->GetDevicePtr<uchar4>();
        auto size = m_OutputBuffer->GetSize();

        a_Width = size.x;
        a_Height = size.y;

        std::vector<uint8_t> pixels;
        pixels.resize(size.x * size.y * sizeof(uchar4));

        cudaMemcpy(pixels.data(), devPtr, pixels.size(), cudaMemcpyKind::cudaMemcpyDeviceToHost);

        return pixels;
    }

    void WaveFrontRenderer::ResizeInteropTexture(
        const std::unique_ptr<InteropGPUTexture>& a_InteropTexture, 
        Microsoft::WRL::ComPtr<ID3D11Texture2D>& a_TextureResource,
        const uint3& a_NewSize) const
    {
        //First unmap the resource, to be sure that it is not mapped anymore as texture resource will be recreated.
        a_InteropTexture->Unmap();

        //Unregister the resource for use by Cuda as the texture resource will be recreated.
        InteropGPUTexture::UnRegisterResource(a_TextureResource);

        //Resize the DX11 texture resource, will delete and create a new texture with the same ComPtr.
        m_DX11Wrapper->ResizeTexture2D(a_TextureResource, a_NewSize);

        //Register the new texture for use by Cuda.
        a_InteropTexture->RegisterResource(a_TextureResource);

        //Map the resource to be used by Cuda after this call.
        a_InteropTexture->Map();

        //Initialize the buffer with 0s.
        a_InteropTexture->Clear();

        //Unmap the resource to no longer be used by Cuda after this call.
        a_InteropTexture->Unmap();

    }

    void WaveFrontRenderer::ResizeBuffers()
    {
        printf("\n\nRESIZING WAVEFRONT BUFFERS!!\n\n");
    	
        CHECKLASTCUDAERROR;

        ////Set up the OpenGL output buffer.
        //m_OutputBuffer->Resize(m_Settings.outputResolution.x, m_Settings.outputResolution.y);

        // TODO: num pixels using render resolution
        //Set up buffers.
        const unsigned numPixels = m_Settings.outputResolution.x * m_Settings.outputResolution.y;

        //CheckCudaLastErr();
        m_IntermediateOutputBuffer.Resize(sizeof(uchar4) * numPixels);

        

        {//Resize pixel buffers.
            //Multiple channel pixel buffer.
            //Unmap the resources to be sure they aren't being used anymore.
            for (auto& channel : m_PixelBufferSeparate)
            {
                channel->Unmap();
            }

            //First unregister all resources as the original d3d11 resource will get deleted and re-created.
            InteropGPUTexture::UnRegisterResource(m_D3D11PixelBufferSeparate);

            m_DX11Wrapper->ResizeTexture2D(m_D3D11PixelBufferSeparate, { m_Settings.renderResolution.x, m_Settings.renderResolution.y, s_numLightChannels });

            for (unsigned int channelIndex = 0; channelIndex < s_numLightChannels; ++channelIndex)
            {

                m_PixelBufferSeparate[channelIndex]->RegisterResource(m_D3D11PixelBufferSeparate);

                //Make sure to initialize the buffer with 0s
                m_PixelBufferSeparate[channelIndex]->Map(channelIndex);
                m_PixelBufferSeparate[channelIndex]->Clear();
                m_PixelBufferSeparate[channelIndex]->Unmap();

            }

            //Single channel pixel buffer.
            m_PixelBufferCombined->Unmap();
            InteropGPUTexture::UnRegisterResource(m_D3D11PixelBufferCombined);
            m_DX11Wrapper->ResizeTexture2D(m_D3D11PixelBufferCombined, { m_Settings.renderResolution.x, m_Settings.renderResolution.y, 1 });

            //Single channel pixel buffer.
            ResizeInteropTexture(m_PixelBufferCombined, m_D3D11PixelBufferCombined, {m_Settings.renderResolution.x, m_Settings.renderResolution.y, 1});
        }

        //Single channel upscaled output buffer
        m_PixelBufferUpscaled->Unmap();
        InteropGPUTexture::UnRegisterResource(m_D3D11PixelBufferUpscaled);
        m_DX11Wrapper->ResizeTexture2D(m_D3D11PixelBufferUpscaled, { m_Settings.outputResolution.x, m_Settings.outputResolution.y, 1 });

        m_PixelBufferUpscaled->RegisterResource(m_D3D11PixelBufferUpscaled);
        m_PixelBufferUpscaled->Map();
        m_PixelBufferUpscaled->Clear();
        m_PixelBufferUpscaled->Unmap();

        //Depth buffer.
        ResizeInteropTexture(m_DepthBuffer, m_D3D11DepthBuffer, { m_Settings.renderResolution.x, m_Settings.renderResolution.y, 1 });

        //Jitter buffer.
        ResizeInteropTexture(m_JitterBuffer, m_D3D11JitterBuffer, { m_Settings.renderResolution.x, m_Settings.renderResolution.y, 1 });

        //Motion vector buffer.
        ResizeInteropTexture(m_MotionVectorBuffer, m_D3D11MotionVectorBuffer, { m_Settings.renderResolution.x, m_Settings.renderResolution.y, 1 });

        //Normal-Roughness buffer.
        ResizeInteropTexture(m_NormalRoughnessBuffer, m_D3D11NormalRoughnessBuffer, { m_Settings.renderResolution.x, m_Settings.renderResolution.y, 1 });
        
        //Initialize the ray buffers. Note: These are not initialized but Reset() is called when the waves start.
        const auto numPrimaryRays = numPixels;
        const auto numShadowRays = numPixels * m_Settings.depth;// +(numPixels * ReSTIRSettings::numReservoirsPerPixel); //TODO: change to 2x num pixels and add safety check to resolve when full.

        //Create atomic buffers. This automatically sets the counter to 0 and size to max.
        CreateAtomicBuffer<IntersectionRayData>(&m_Rays, numPrimaryRays);
        CreateAtomicBuffer<ShadowRayData>(&m_ShadowRays, numShadowRays);
		CreateAtomicBuffer<ShadowRayData>(&m_VolumetricShadowRays, numShadowRays * 5);	//TODO: replace hardcoded number of samples (5)
		CreateAtomicBuffer<IntersectionData>(&m_IntersectionData, numPixels);
		CreateAtomicBuffer<VolumetricIntersectionData>(&m_VolumetricIntersectionData, numPixels);

		//Initialize each surface data buffer.
		for (auto& surfaceDataBuffer : m_SurfaceData)
		{
			//Note; Only allocates memory and stores the size on the GPU. It does not actually fill any data in yet.
			surfaceDataBuffer.Resize(numPixels * sizeof(SurfaceData));
		}

		for (auto& volumetricDataBuffer : m_VolumetricData)
		{
			volumetricDataBuffer.Resize(numPixels * sizeof(VolumetricData));
		}

        //Use mostly the default values.
        ReSTIRSettings rSettings;
        rSettings.width = m_Settings.renderResolution.x;
        rSettings.height = m_Settings.renderResolution.y;
        // A null frame snapshot will not record anything when requested to.   

        m_ReSTIR = std::make_unique<ReSTIR>();

        //Print the expected VRam usage.
        size_t requiredSize = m_ReSTIR->GetExpectedGpuRamUsage(rSettings, 3);
        printf("Initializing ReSTIR. Expected VRam usage in bytes: %llu\n", requiredSize);

        CHECKLASTCUDAERROR;
        //Finally actually allocate memory for ReSTIR.
        m_ReSTIR->Initialize(rSettings);


        size_t usedSize = m_ReSTIR->GetAllocatedGpuMemory();
        printf("Actual bytes allocated by ReSTIR: %llu\n", usedSize);
    }

    void WaveFrontRenderer::SetOutputResolutionInternal(glm::uvec2 a_NewResolution)
    {
        m_IntermediateSettings.outputResolution.x = a_NewResolution.x;
        m_IntermediateSettings.outputResolution.y = a_NewResolution.y;
    }

    void WaveFrontRenderer::WaitForDeferredCalls()
    {
        if (!m_DeferredOpenGLCalls.empty())
        {            
            std::mutex mutex;
            std::unique_lock lock(mutex);

            m_OGLCallCondition.wait(lock, [this]()
            {
                    return m_DeferredOpenGLCalls.empty();
            });
        }
    }

    void WaveFrontRenderer::FinalizeFrameStats()
    {
        std::lock_guard lk(m_FrameStatsMutex);

        m_LastFrameStats = m_CurrentFrameStats;
        m_LastFrameStats.m_Id++;
        m_CurrentFrameStats = {};
        m_CurrentFrameStats.m_Id = m_LastFrameStats.m_Id;
    }
}
#endif