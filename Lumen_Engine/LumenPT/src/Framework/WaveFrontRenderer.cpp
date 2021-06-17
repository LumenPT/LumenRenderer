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
#include "../CUDAKernels/WaveFrontKernels/EmissiveLookup.cuh"
#include "../Shaders/CppCommon/WaveFrontDataStructs.h"
#include "CudaUtilities.h"
#include "ReSTIR.h"
#include "../Tools/FrameSnapshot.h"
#include "../Tools/SnapShotProcessing.cuh"
#include "MotionVectors.h"
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

        


        //m_DX11Wrapper->GetDevice();

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

        //m_NRD = std::make_unique<NrdWrapper>();
        //NRDWrapperInitParams nrdInitParams;
        //nrdInitParams.m_InputImageWidth = m_Settings.renderResolution.x;
        //nrdInitParams.m_InputImageHeight = m_Settings.renderResolution.y;
        //nrdInitParams.m_pServiceLocator = &m_ServiceLocator;
        //m_NRD->Initialize(nrdInitParams);

        //m_OptixDenoiser = std::make_unique<OptixDenoiserWrapper>();
        //OptixDenoiserInitParams optixDenoiserInitParams;
        //optixDenoiserInitParams.m_InputWidth = m_Settings.renderResolution.x;
        //optixDenoiserInitParams.m_InputHeight = m_Settings.renderResolution.y;
        //optixDenoiserInitParams.m_ServiceLocator = &m_ServiceLocator;
        //m_OptixDenoiser->Initialize(optixDenoiserInitParams);

        m_OptixDenoiser = std::make_unique<OptixDenoiserWrapper>();
        OptixDenoiserInitParams optixDenoiserInitParams;
        // TODO: Optix Denoiers render resolution input
        optixDenoiserInitParams.m_InputWidth = m_Settings.renderResolution.x;
        optixDenoiserInitParams.m_InputHeight = m_Settings.renderResolution.y;
        optixDenoiserInitParams.m_ServiceLocator = &m_ServiceLocator;
        m_OptixDenoiser->Initialize(optixDenoiserInitParams);

        //Set up the OpenGL output buffer.
        m_OutputBuffer = std::make_unique<CudaGLTexture>(GL_RGBA8, m_Settings.outputResolution.x, m_Settings.outputResolution.y, 4);
        SetRenderResolution(glm::uvec2(m_Settings.outputResolution.x, m_Settings.outputResolution.y));

        //Set up buffers.

        //Set up pixel output buffers.

        // TODO: DX11 Pixel Buffer render resolution input
        m_D3D11PixelBufferSeparate = m_DX11Wrapper->CreateTexture2D({ m_Settings.renderResolution.x, m_Settings.renderResolution.y, s_numLightChannels});

        m_D3D11PixelBufferCombined = m_DX11Wrapper->CreateTexture2D({ m_Settings.renderResolution.x, m_Settings.renderResolution.y, 1 });

        cudaTextureDesc pixelBufferDesc {};
        memset(&pixelBufferDesc, 0, sizeof(pixelBufferDesc));

        pixelBufferDesc.addressMode[0] = cudaAddressModeClamp;
        pixelBufferDesc.addressMode[0] = cudaAddressModeClamp;
        pixelBufferDesc.filterMode = cudaFilterModePoint;
        pixelBufferDesc.readMode = cudaReadModeElementType;

        for(unsigned int channelIndex = 0; channelIndex < s_numLightChannels; ++channelIndex)
        {
            m_PixelBufferSeparate[channelIndex] = 
                std::make_unique<InteropGPUTexture>(
                    m_D3D11PixelBufferSeparate, 
                    pixelBufferDesc, 
                    cudaGraphicsRegisterFlagsSurfaceLoadStore);

            m_PixelBufferSeparate[channelIndex]->Map(channelIndex);
            m_PixelBufferSeparate[channelIndex]->Clear();
            m_PixelBufferSeparate[channelIndex]->Unmap();
        }

        m_PixelBufferCombined =
            std::make_unique<InteropGPUTexture>(
                m_D3D11PixelBufferCombined,
                pixelBufferDesc,
                cudaGraphicsRegisterFlagsSurfaceLoadStore);

        m_PixelBufferCombined->Map();
        m_PixelBufferCombined->Clear();
        m_PixelBufferCombined->Unmap();

        ResizeBuffers();

        CreateAtomicBuffer<WaveFront::TriangleLight>(&m_TriangleLights, 1000000);

        //Set the service locator pointer to point to the m'table.
        /*m_Table = std::make_unique<SceneDataTable>();
        m_ServiceLocator.m_SceneDataTable = m_Table.get();*/
        CHECKLASTCUDAERROR;

        m_ServiceLocator.m_Renderer = this;
        m_FrameSnapshot = std::make_unique<NullFrameSnapshot>();

        m_ModelConverter.SetRendererRef(*this);

        /*m_DLSS = std::make_unique<DlssWrapper>();
        DLSSWrapperInitParams dlssInitParams;
        dlssInitParams.m_InputImageWidth = m_Settings.renderResolution.x;
        dlssInitParams.m_InputImageHeight = m_Settings.renderResolution.y;
        dlssInitParams.m_OutputImageWidth = m_Settings.outputResolution.x;
        dlssInitParams.m_OutputImageHeight = m_Settings.outputResolution.y;
        dlssInitParams.m_pServiceLocator = &m_ServiceLocator;
        m_DLSS->Initialize(dlssInitParams);*/

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
        SetOutputResolutionInternal(a_NewResolution);
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
        CHECKLASTCUDAERROR;

        //Retrieve the acceleration structure and scene data table once.
        m_OptixSystem->UpdateSBT();
        CHECKLASTCUDAERROR;
        auto begin = std::chrono::high_resolution_clock::now();

        auto* sceneDataTableAccessor = static_cast<PTScene*>(m_Scene.get())->GetSceneDataTableAccessor();

        auto end = std::chrono::high_resolution_clock::now();

        auto milli = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        printf("\n\nTime elapsed to build scene data table: %li milliseconds\n\n", milli);

        CHECKLASTCUDAERROR;

        auto accelerationStructure = std::static_pointer_cast<PTScene>(m_Scene)->GetSceneAccelerationStructure();
        cudaDeviceSynchronize();
        CHECKLASTCUDAERROR;

        //Timer to measure how long each frame takes.
        Timer timer;
        ResetAtomicBuffer<WaveFront::TriangleLight>(&m_TriangleLights);
        auto lightBuffer = m_TriangleLights.GetDevicePtr<AtomicBuffer<WaveFront::TriangleLight>>();

        for (auto& meshInstance : m_Scene->m_MeshInstances)
        {
            //Only run when emission is not disabled, and override is active OR the GLTF has specified valid emissive triangles and mode is set to ENABLED.
            if (meshInstance->GetEmissionMode() != Lumen::EmissionMode::DISABLED 
                && ((meshInstance->GetMesh()->GetEmissiveness() && meshInstance->GetEmissionMode() == Lumen::EmissionMode::ENABLED)
                || meshInstance->GetEmissionMode() == Lumen::EmissionMode::OVERRIDE))
            {
                PTMeshInstance* asPTInstance = reinterpret_cast<PTMeshInstance*>(meshInstance.get());

                //Loop over all instances.

                for (auto& prim : meshInstance->GetMesh()->m_Primitives)
                {
                    auto ptPrim = static_cast<PTPrimitive*>(prim.get());

                    //Find the primitive instance in the data table.
                    auto& entryMap = asPTInstance->GetInstanceEntryMap();
                    auto entry = &entryMap.at(prim.get());

                    AddToLightBufferWrap(
                        ptPrim->m_VertBuffer->GetDevicePtr<Vertex>(),
                        ptPrim->m_IndexBuffer->GetDevicePtr<uint32_t>(),
                        ptPrim->m_BoolBuffer->GetDevicePtr<bool>(),
                        ptPrim->m_IndexBuffer->GetSize() / sizeof(uint32_t),
                        lightBuffer,
                        sceneDataTableAccessor,
                        entry->m_TableIndex);
                }
                
            }
            else
            {
                continue;
            }
        }

        //Don't render if there is no light in the scene as everything will be black anyway.
        const unsigned int numLightsInScene = GetAtomicCounter<WaveFront::TriangleLight>(&m_TriangleLights);
        if (numLightsInScene == 0)
        {
            // Are you sure there are lights in the scene? Then restart the application, weird bugs man.
            __debugbreak();
            return;
        }

        bool recordingSnapshot = m_StartSnapshot;
        if (m_StartSnapshot)
        {
            // Replacing the snapshot with a non-null one will start recording requested features.
            m_FrameSnapshot = std::make_unique<FrameSnapshot>();
            m_StartSnapshot = false;
        }

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

        for(auto& buffer : m_PixelBufferSeparate)
        {
            buffer->Clear();   
        }

        //Only clean the merged buffer if no blending is enabled.
        if (!m_Settings.blendOutput)
        {
            m_PixelBufferCombined->Clear();
        }

        cudaDeviceSynchronize();
        CHECKLASTCUDAERROR;

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
        GeneratePrimaryRays(primaryRayGenParams);
        cudaDeviceSynchronize();
        CHECKLASTCUDAERROR;

        //Set the atomic counter for primary rays to the amount of pixels.
        SetAtomicCounter<IntersectionRayData>(&m_Rays, numPixels);

        //##ToolsBookmark
        //Example of defining how to add buffers to the pixel debugger tool
        //This lambda is NOT ran every frame, it is only ran when the output layer requests a snapshot
        m_FrameSnapshot->AddBuffer([&]()
            {
                // The buffers that need to be given to the tool are provided via a map as shown below
                // Notice that CudaGLTextures are used, as opposed to memory buffers. This is to enable the data to be used with OpenGL
                // and thus displayed via ImGui
                std::map<std::string, FrameSnapshot::ImageBuffer> resBuffers;

                // TODO: render resolution used for snapshots
                m_DeferredOpenGLCalls.push([&]() {
                    resBuffers["Origins"].m_Memory = std::make_unique<CudaGLTexture>(GL_RGB32F, m_Settings.renderResolution.x,
                        m_Settings.renderResolution.y, 3 * sizeof(float));

                    resBuffers["Directions"].m_Memory = std::make_unique<CudaGLTexture>(GL_RGB32F, m_Settings.renderResolution.x,
                        m_Settings.renderResolution.y, 3 * sizeof(float));

                    resBuffers["Contributions"].m_Memory = std::make_unique<CudaGLTexture>(GL_RGB32F, m_Settings.renderResolution.x,
                        m_Settings.renderResolution.y, 3 * sizeof(float));
                    });

                WaitForDeferredCalls();
                // A CUDA kernel used to separate the interleave primary ray buffer into 3 different buffers
                // This is the main reason we use a lambda, as it needs to be defined how to interpret the data
                SeparateIntersectionRayBufferCPU((m_Rays.GetSize() - sizeof(uint32_t)) / sizeof(IntersectionRayData),
                    m_Rays.GetDevicePtr<AtomicBuffer<IntersectionRayData>>(),
                    m_Settings.renderResolution,                            // TODO: render resolution used for intersection ray buffer separation
                    resBuffers.at("Origins").m_Memory->GetDevicePtr<float3>(),
                    resBuffers.at("Directions").m_Memory->GetDevicePtr<float3>(),
                    resBuffers.at("Contributions").m_Memory->GetDevicePtr<float3>());

                return resBuffers;
            });

        // TODO: render resolution used in snapshot->AddBuffer
        m_FrameSnapshot->AddBuffer([&]()
            {
                auto motionVectorBuffer = m_MotionVectors.GetMotionVectorBuffer();

                std::map<std::string, FrameSnapshot::ImageBuffer> resBuffers;
                m_DeferredOpenGLCalls.push([&]() {
                    resBuffers["Motion vector direction"].m_Memory = std::make_unique<CudaGLTexture>(GL_RGB32F, m_Settings.renderResolution.x,
                        m_Settings.renderResolution.y, 3 * sizeof(float));

                    resBuffers["Motion vector magnitude"].m_Memory = std::make_unique<CudaGLTexture>(GL_RGB32F, m_Settings.renderResolution.x,
                        m_Settings.renderResolution.y, 3 * sizeof(float));
                    });
                WaitForDeferredCalls();

                SeparateMotionVectorBufferCPU(m_Settings.renderResolution.x * m_Settings.renderResolution.y,
                    motionVectorBuffer->GetDevicePtr<MotionVectorBuffer>(),
                    resBuffers.at("Motion vector direction").m_Memory->GetDevicePtr<float3>(),
                    resBuffers.at("Motion vector magnitude").m_Memory->GetDevicePtr<float3>()
                );

                return resBuffers;
            });

        // TODO: render resolution used in snapshot->AddBuffer
        m_FrameSnapshot->AddBuffer([&]()
            {
                auto optixDenoiserInput = &(m_OptixDenoiser->TestInput);
                auto optixDenoiszerOutput = &(m_OptixDenoiser->TestOutput);

                std::map<std::string, FrameSnapshot::ImageBuffer> resBuffers;
                m_DeferredOpenGLCalls.push([&]() {
                    resBuffers["Optix denoiser input"].m_Memory = std::make_unique<CudaGLTexture>(GL_RGB32F, m_Settings.renderResolution.x,
                        m_Settings.renderResolution.y, 3 * sizeof(float));

                    resBuffers["Optix denoiser output"].m_Memory = std::make_unique<CudaGLTexture>(GL_RGB32F, m_Settings.renderResolution.x,
                        m_Settings.renderResolution.y, 3 * sizeof(float));
                    });
                WaitForDeferredCalls();

                SeparateOptixDenoiserBufferCPU(m_Settings.renderResolution.x * m_Settings.renderResolution.y,
                    optixDenoiserInput->GetDevicePtr<float3>(),
                    optixDenoiszerOutput->GetDevicePtr<float3>(),
                    resBuffers.at("Optix denoiser input").m_Memory->GetDevicePtr<float3>(),
                    resBuffers.at("Optix denoiser output").m_Memory->GetDevicePtr<float3>()
                );

                return resBuffers;
            });

        //Clear the surface data that contains information from the second last frame so that it can be reused by this frame.
        cudaMemset(m_SurfaceData[currentIndex].GetDevicePtr(), 0, sizeof(SurfaceData) * numPixels);
        cudaDeviceSynchronize();
        CHECKLASTCUDAERROR;

        //Set the counters back to 0 for intersections and shadow rays.
        const unsigned counterDefault = 0;
        SetAtomicCounter<ShadowRayData>(&m_ShadowRays, counterDefault);
        SetAtomicCounter<IntersectionData>(&m_IntersectionData, counterDefault);
		SetAtomicCounter<VolumetricIntersectionData>(&m_VolumetricIntersectionData, counterDefault);
        CHECKLASTCUDAERROR;

        //Pass the buffers to the optix shader for shading.
        OptixLaunchParameters rayLaunchParameters {};
        rayLaunchParameters.m_TraceType = RayType::INTERSECTION_RAY;
        rayLaunchParameters.m_MinMaxDistance = { 0.01f, 5000.f };
        rayLaunchParameters.m_IntersectionBuffer = m_IntersectionData.GetDevicePtr<AtomicBuffer<IntersectionData>>();
		rayLaunchParameters.m_VolumetricIntersectionBuffer = m_VolumetricIntersectionData.GetDevicePtr<AtomicBuffer<VolumetricIntersectionData>>();
        rayLaunchParameters.m_IntersectionRayBatch = m_Rays.GetDevicePtr<AtomicBuffer<IntersectionRayData>>();
        rayLaunchParameters.m_SceneData = sceneDataTableAccessor;
        rayLaunchParameters.m_TraversableHandle = accelerationStructure;
        // TODO: render resolution used in ray launch params
        rayLaunchParameters.m_ResolutionAndDepth = uint3{ m_Settings.renderResolution.x, m_Settings.renderResolution.y, m_Settings.depth };

        //Set the amount of rays to trace. Initially same as screen size.
        auto numIntersectionRays = numPixels;

        auto seed = WangHash(frameCount);

        /*
         * Resolve rays and shade at every depth.
         */
        for (unsigned depth = 0; depth < m_Settings.depth && numIntersectionRays > 0; ++depth)
        {
            //Tell Optix to resolve the primary rays that have been generated.
            m_OptixSystem->TraceRays(numIntersectionRays, rayLaunchParameters);
            cudaDeviceSynchronize();
            CHECKLASTCUDAERROR;

            /*
             * Calculate the surface data for this depth.
             */
            unsigned numIntersections = 0;
            numIntersections = GetAtomicCounter<IntersectionData>(&m_IntersectionData);

            //1 and 2 are used for the first intersection and remembered for temporal use.
            const auto surfaceDataBufferIndex = (depth == 0 ? currentIndex : 2);   

            ExtractSurfaceData(
                numIntersections,
                m_IntersectionData.GetDevicePtr<AtomicBuffer<IntersectionData>>(),
                m_Rays.GetDevicePtr<AtomicBuffer<IntersectionRayData>>(),
                m_SurfaceData[surfaceDataBufferIndex].GetDevicePtr<SurfaceData>(),
                m_Settings.renderResolution,        // TODO: render resolution used to extract surface data
                sceneDataTableAccessor);

            unsigned numVolumeIntersections = 0;
            m_VolumetricIntersectionData.Read(&numVolumeIntersections, sizeof(numVolumeIntersections), 0);

            const auto volumetricDataBufferIndex = 0;

            ExtractVolumetricData(
                numVolumeIntersections,
                m_VolumetricIntersectionData.GetDevicePtr<AtomicBuffer<VolumetricIntersectionData>>(),
                m_Rays.GetDevicePtr<AtomicBuffer<IntersectionRayData>>(),
                m_VolumetricData[volumetricDataBufferIndex].GetDevicePtr<VolumetricData>(),
                m_Settings.renderResolution,        // TODO: render resolution used to extract volumetric data
                sceneDataTableAccessor);

            cudaDeviceSynchronize();
            CHECKLASTCUDAERROR;

            //motion vector generation
            if (depth == 0)
            {
                glm::mat4 previousFrameMatrix, currentFrameMatrix;
                m_Scene->m_Camera->GetMatrixData(previousFrameMatrix, currentFrameMatrix);
                sutil::Matrix4x4 prevFrameMatrixArg = ConvertGLMtoSutilMat4(previousFrameMatrix);

                glm::mat4 projectionMatrix = m_Scene->m_Camera->GetProjectionMatrix();
                sutil::Matrix4x4 projectionMatrixArg = ConvertGLMtoSutilMat4(projectionMatrix);

                MotionVectorsGenerationData motionVectorsGenerationData;
                motionVectorsGenerationData.m_MotionVectorBuffer = nullptr;
                motionVectorsGenerationData.a_CurrentSurfaceData = m_SurfaceData[currentIndex].GetDevicePtr<SurfaceData>();
                // TODO: Render resolution used in motion vector generation data
                motionVectorsGenerationData.m_ScreenResolution = make_uint2(m_Settings.renderResolution.x, m_Settings.renderResolution.y);
                motionVectorsGenerationData.m_PrevViewMatrix = prevFrameMatrixArg.inverse();
                motionVectorsGenerationData.m_ProjectionMatrix = projectionMatrixArg;
                m_MotionVectors.Update(motionVectorsGenerationData);
            }

            /*
             * Call the shading kernels.
             */

            ShadingLaunchParameters shadingLaunchParams(
                uint3{ m_Settings.renderResolution.x, m_Settings.renderResolution.y, m_Settings.depth },
                m_SurfaceData[surfaceDataBufferIndex].GetDevicePtr<SurfaceData>(),
                m_SurfaceData[temporalIndex].GetDevicePtr<SurfaceData>(),
                m_VolumetricData[volumetricDataBufferIndex].GetDevicePtr<VolumetricData>(),
                m_MotionVectors.GetMotionVectorBuffer()->GetDevicePtr<MotionVectorBuffer>(),
                &m_TriangleLights,
                accelerationStructure,
                m_OptixSystem.get(),
                depth,
                seed,
                m_Rays.GetDevicePtr<AtomicBuffer<IntersectionRayData>>(),
                m_ShadowRays.GetDevicePtr<AtomicBuffer<ShadowRayData>>(),
                m_VolumetricShadowRays.GetDevicePtr<AtomicBuffer<ShadowRayData>>(),
                m_ReSTIR.get(),
                pixelBuffers);

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

        //The amount of shadow rays to trace.
        unsigned numShadowRays = GetAtomicCounter<ShadowRayData>(&m_ShadowRays);

        if (numShadowRays > 0)
        {

            

            //The settings for shadow ray resolving.
            OptixLaunchParameters shadowRayLaunchParameters;
            shadowRayLaunchParameters = rayLaunchParameters; //Copy settings from the intersection rays.
            shadowRayLaunchParameters.m_TraceType = RayType::SHADOW_RAY;
            shadowRayLaunchParameters.m_OutputChannels = pixelBuffers;
            shadowRayLaunchParameters.m_ShadowRayBatch = m_ShadowRays.GetDevicePtr<AtomicBuffer<ShadowRayData>>();

            //Tell optix to resolve the shadow rays.
            m_OptixSystem->TraceRays(numShadowRays, shadowRayLaunchParameters);
            cudaDeviceSynchronize();
            CHECKLASTCUDAERROR;
        }

        //Post-processing
        {
            // TODO: render and output resolution used in post processing
            PostProcessLaunchParameters postProcessLaunchParams(
                m_Settings.renderResolution,
                m_Settings.outputResolution,
                pixelBuffers,
                m_PixelBufferCombined->GetSurfaceObject(),
                m_IntermediateOutputBuffer.GetDevicePtr<uchar4>(),
                m_Settings.blendOutput,
                m_BlendCounter
            );
            //Post processing using CUDA kernel.
            //PostProcess(postProcessLaunchParams);

            MergeOutput(postProcessLaunchParams);
            cudaDeviceSynchronize();
            CHECKLASTCUDAERROR;

            OptixDenoiserLaunchParameters optixDenoiserLaunchParams(
                m_Settings.renderResolution,
                m_PixelBufferCombined->GetSurfaceObject(),
                m_OptixDenoiser->TestInput.GetDevicePtr<float3>(),
                m_OptixDenoiser->TestOutput.GetDevicePtr<float3>()
            );


            //PrepareOptixDenoising(optixDenoiserLaunchParams);
            //CHECKLASTCUDAERROR;

            //OptixDenoiserDenoiseParams optixDenoiserParams = {};
            //optixDenoiserParams.m_PostProcessLaunchParams = &postProcessLaunchParams;
            ///*optixDenoiserParams.m_ColorInput = m_IntermediateOutputBuffer.GetCUDAPtr();
            //optixDenoiserParams.m_Output = m_IntermediateOutputBuffer.GetCUDAPtr();*/
            //optixDenoiserParams.m_ColorInput = m_OptixDenoiser->TestInput.GetCUDAPtr();
            //optixDenoiserParams.m_Output = m_OptixDenoiser->TestOutput.GetCUDAPtr();
            //m_OptixDenoiser->Denoise(optixDenoiserParams);
            ////cudaDeviceSynchronize();
            //CHECKLASTCUDAERROR;

            //FinishOptixDenoising(optixDenoiserLaunchParams);
            //CHECKLASTCUDAERROR;

            WriteToOutput(postProcessLaunchParams);
            cudaDeviceSynchronize();
            CHECKLASTCUDAERROR;

            // TODO: Render and output resolutions used in DLSS init params
            //kinda gross to reinitialize dlssInitParams here again but ohwell
            DLSSWrapperInitParams dlssInitParams;
            dlssInitParams.m_InputImageWidth = m_Settings.renderResolution.x;
            dlssInitParams.m_InputImageHeight = m_Settings.renderResolution.y;
            dlssInitParams.m_OutputImageWidth = m_Settings.renderResolution.x * 2;
            dlssInitParams.m_OutputImageHeight = m_Settings.renderResolution.y * 2;
            //dlssInitParams.m_OutputImageWidth = m_Settings.outputResolution.x;
            //dlssInitParams.m_OutputImageHeight = m_Settings.outputResolution.y;
            dlssInitParams.m_pServiceLocator = &m_ServiceLocator;

            m_PixelBufferCombined->Unmap();

            //m_DX11Wrapper->GetContext()->CopyResource(m_DX11Wrapper->m_D3D11PixelBufferCombined, m_D3D11PixelBufferCombined.Get());

            m_DLSS->EvaluateDLSS(dlssInitParams, m_D3D11PixelBufferCombined, m_MotionVectors.GetMotionVectorDirectionsTex());

            //construct motion vector tex 
            //std::shared_ptr<InteropGPUTexture>();
        }

        //Cuda should no longer be operating on the pixelbuffers.
        for(auto& buffer : m_PixelBufferSeparate)
        {
            buffer->Unmap();
        }

        //m_PixelBufferCombined->Unmap();

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


        // TODO: Weird debug code. Yeet?
        m_DebugTexture = m_OutputTexture;
        //#if defined(_DEBUG)
        /*m_MotionVectors.GenerateDebugTextures();
        m_DebugTexture = m_MotionVectors.GetMotionVectorMagnitudeTex();*/

        /*m_OptixDenoiser->UpdateDebugTextures();
        m_DebugTexture = m_OptixDenoiser->m_OptixDenoiserInputTex.m_Memory->GetTexture();*/

        m_SnapshotReady = recordingSnapshot;
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

            FindEmissivesWrap(
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

        std::unique_ptr<PTPrimitive> prim = std::make_unique<PTPrimitive>(std::move(vertexBuffer), std::move(indexBuffer), std::move(emissiveBuffer), std::move(gAccel));
        prim->m_Material = a_PrimitiveData.m_Material;
        prim->m_ContainEmissive = numLights > 0 ? true : false;
        prim->m_NumLights = numLights;

        prim->m_DevicePrimitive.m_VertexBuffer = prim->m_VertBuffer->GetDevicePtr<Vertex>();
        prim->m_DevicePrimitive.m_IndexBuffer = prim->m_IndexBuffer->GetDevicePtr<unsigned int>();
        prim->m_DevicePrimitive.m_Material = std::static_pointer_cast<PTMaterial>(prim->m_Material)->GetDeviceMaterial();
        prim->m_DevicePrimitive.m_IsEmissive = prim->m_BoolBuffer->GetDevicePtr<bool>();
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
        uint32_t a_Width, uint32_t a_Height)
    {
        static cudaChannelFormatDesc formatDesc = cudaCreateChannelDesc<uchar4>();
        return std::make_shared<PTTexture>(a_PixelData, formatDesc, a_Width, a_Height);
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
        dlssInitParams.m_pServiceLocator = &m_ServiceLocator;
        m_DLSS->Initialize(dlssInitParams);

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
        auto devPtr = m_OutputBuffer->GetDevicePtr<uchar4>();
        auto size = m_OutputBuffer->GetSize();

        a_Width = size.x;
        a_Height = size.y;

        std::vector<uint8_t> pixels;
        pixels.resize(size.x * size.y * sizeof(uchar4));
        std::lock_guard<std::mutex> lock(m_OutputBufferMutex);

        cudaMemcpy(pixels.data(), devPtr, pixels.size(), cudaMemcpyKind::cudaMemcpyDeviceToHost);

        return pixels;
    }

    void WaveFrontRenderer::ResizeBuffers()
    {
        printf("\n\nRESIZING WAVEFRONT BUFFERS!!\n\n");
    	
        CHECKLASTCUDAERROR;

        ////Set up the OpenGL output buffer.
        //m_OutputBuffer->Resize(m_Settings.outputResolution.x, m_Settings.outputResolution.y);

        // TODO: num pixels using render resolution
        //Set up buffers.
        const unsigned numPixels = m_Settings.renderResolution.x * m_Settings.renderResolution.y;
        const unsigned numOutputChannels = static_cast<unsigned>(LightChannel::NUM_CHANNELS);

        //CheckCudaLastErr();
        m_IntermediateOutputBuffer.Resize(sizeof(uchar4) * numPixels);

        

        //Allocate pixel buffer.
        //TODO: m_PixelBufferSeparate->Resize({ m_Settings.renderResolution.x, m_Settings.renderResolution.y, numOutputChannels });

        //Single channel pixel buffer.
        //TODO: m_PixelBufferCombined->Resize({m_Settings.renderResolution.x, m_Settings.renderResolution.y, 0});

        //Initialize the ray buffers. Note: These are not initialized but Reset() is called when the waves start.
        const auto numPrimaryRays = numPixels;
        const auto numShadowRays = numPixels * m_Settings.depth + (numPixels * ReSTIRSettings::numReservoirsPerPixel); //TODO: change to 2x num pixels and add safety check to resolve when full.

        //Create atomic buffers. This automatically sets the counter to 0 and size to max.
        CreateAtomicBuffer<IntersectionRayData>(&m_Rays, numPrimaryRays);
        CreateAtomicBuffer<ShadowRayData>(&m_ShadowRays, numShadowRays);
		CreateAtomicBuffer<ShadowRayData>(&m_VolumetricShadowRays, numShadowRays);
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

        // TODO: motion vectors initialized with render resolution
        m_MotionVectors.Init(make_uint2(m_Settings.renderResolution.x, m_Settings.renderResolution.y));

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

}
#endif