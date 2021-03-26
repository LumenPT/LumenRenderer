#if defined(WAVEFRONT)

#include "WaveFrontRenderer.h"
#include "PTPrimitive.h"
#include "PTMesh.h"
#include "PTScene.h"
#include "Material.h"
#include "Texture.h"
#include "PTVolume.h"
#include "Material.h"
#include "MemoryBuffer.h"
#include "CudaGLTexture.h"
#include "SceneDataTable.h"
#include "../CUDAKernels/WaveFrontKernels.cuh"
#include "../CUDAKernels/WaveFrontKernels/EmissiveLookup.cuh"
#include "../Shaders/CppCommon/LumenPTConsts.h"
#include "../Shaders/CppCommon/WaveFrontDataStructs.h"
#include "CudaUtilities.h"
#include "ReSTIR.h"
#include "../Tools/FrameSnapshot.h"
#include "../Tools/SnapShotProcessing.cuh"
#include "MotionVectors.h"

#include <Optix/optix_function_table_definition.h>
#include <filesystem>
#include <glm/gtx/compatibility.hpp>
#include <sutil/Matrix.h>

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
        m_FrameIndex = 0;
        m_Settings = a_Settings;

        //Init CUDA
        cudaFree(0);
        m_CUDAContext = 0;
        

        //TODO: Ensure shader names match what we put down here.
        OptixWrapper::InitializationData optixInitData;
        optixInitData.m_CUDAContext = m_CUDAContext;
        optixInitData.m_ProgramData.m_ProgramPath = LumenPTConsts::gs_ShaderPathBase + "WaveFrontShaders.ptx";;
        optixInitData.m_ProgramData.m_ProgramLaunchParamName = "launchParams";
        optixInitData.m_ProgramData.m_ProgramRayGenFuncName = "__raygen__WaveFrontRG";
        optixInitData.m_ProgramData.m_ProgramMissFuncName = "__miss__WaveFrontMS";
        optixInitData.m_ProgramData.m_ProgramAnyHitFuncName = "__anyhit__WaveFrontAH";
        optixInitData.m_ProgramData.m_ProgramClosestHitFuncName = "__closesthit__WaveFrontCH";
        optixInitData.m_ProgramData.m_MaxNumHitResultAttributes = 2;
        optixInitData.m_ProgramData.m_MaxNumPayloads = 2;

        m_OptixSystem = std::make_unique<OptixWrapper>(optixInitData);

        //Set the service locator's pointer to the OptixWrapper.
        m_ServiceLocator.m_OptixWrapper = m_OptixSystem.get();

        //Set up the OpenGL output buffer.
        m_OutputBuffer.Resize(m_Settings.outputResolution.x, m_Settings.outputResolution.y);

        //Set up buffers.
        const unsigned numPixels = m_Settings.renderResolution.x * m_Settings.renderResolution.y;
        const unsigned numOutputChannels = static_cast<unsigned>(LightChannel::NUM_CHANNELS);

        //Allocate pixel buffer.
        m_PixelBufferSeparate.Resize(sizeof(float3) * numPixels * numOutputChannels);

        //Single channel pixel buffer.
        m_PixelBufferCombined.Resize(sizeof(float3) * numPixels);

        //Initialize the ray buffers. Note: These are not initialized but Reset() is called when the waves start.
        const auto numPrimaryRays = numPixels;
        const auto numShadowRays = numPixels * m_Settings.depth + (numPixels * ReSTIRSettings::numReservoirsPerPixel); //TODO: change to 2x num pixels and add safety check to resolve when full.

        //Create atomic buffers. This automatically sets the counter to 0 and size to max.
        CreateAtomicBuffer<IntersectionRayData>(&m_Rays, numPrimaryRays);
        CreateAtomicBuffer<ShadowRayData>(&m_ShadowRays, numShadowRays);
        CreateAtomicBuffer<IntersectionData>(&m_IntersectionData, numPixels);

        //Initialize each surface data buffer.
        for(int i = 0; i < 3; ++i)
        {
            //Note; Only allocates memory and stores the size on the GPU. It does not actually fill any data in yet.
            m_SurfaceData[i].Resize(numPixels * sizeof(SurfaceData));
        }

        //TODO: number of lights will be dynamic per frame but this is temporary.
        constexpr auto numLights = 3;

        m_TriangleLights.Resize(sizeof(TriangleLight) * numLights);

        //Temporary lights, stored in the buffer.
        TriangleLight lights[numLights];

        //Intensity per light.
        lights[0].radiance = { 200000, 200000, 200000 };
        lights[1].radiance = { 200000, 200000, 200000 };
        lights[2].radiance = { 200000, 200000, 200000 };


        //Actually set the triangle lights to have an area.
        lights[0].p0 = {605.f, 700.f, -5.f};
        lights[0].p1 = { 600.f, 700.f, 5.f };
        lights[0].p2 = { 595.f, 700.f, -5.f };

        lights[1].p0 = { 5.f, 700.f, -5.f };
        lights[1].p1 = { 0.f, 700.f, 5.f };
        lights[1].p2 = { -5.f, 700.f, -5.f };

        lights[2].p0 = { -595.f, 700.f, -5.f };
        lights[2].p1 = { -600.f, 700.f, 5.f };
        lights[2].p2 = { -605.f, 700.f, -5.f };

        //Calculate the area per light.
        for(int i = 0; i < 3; ++i)
        {
            float3 vec1 = (lights[i].p0 - lights[i].p1);
            float3 vec2 = (lights[i].p0 - lights[i].p2);
            lights[i].area = sqrt(pow((vec1.y * vec2.z - vec2.y * vec1.z), 2) + pow((vec1.x * vec2.z - vec2.x * vec1.z), 2) + pow((vec1.x * vec2.y - vec2.x * vec1.y), 2)) / 2.f;
            printf("lol");
        }

        //Calculate the normal for each light.
        for(int i = 0; i < 3; ++i)
        {
            glm::vec3 arm1 = normalize(glm::vec3(lights[i].p0.x - lights[i].p2.x, lights[i].p0.y - lights[i].p2.y, lights[i].p0.z - lights[i].p2.z));
            glm::vec3 arm2 = normalize(glm::vec3(lights[i].p0.x - lights[i].p1.x, lights[i].p0.y - lights[i].p1.y, lights[i].p0.z - lights[i].p1.z));
            glm::vec3 normal = normalize(glm::cross(arm2, arm1));
            lights[i].normal = { normal.x, normal.y, normal.z };
        }


        m_TriangleLights.Write(&lights[0], sizeof(TriangleLight) * numLights, 0);

        m_MotionVectors.Init(make_uint2(m_Settings.renderResolution.x, m_Settings.renderResolution.y));
    	
        //Set the service locator pointer to point to the m'table.
        m_Table = std::make_unique<SceneDataTable>();
        m_ServiceLocator.m_SceneDataTable = m_Table.get();

        m_ServiceLocator.m_Renderer = this;

        //Use mostly the default values.
        ReSTIRSettings rSettings;
        rSettings.width = m_Settings.renderResolution.x;
        rSettings.height = m_Settings.renderResolution.y;
        // A null frame snapshot will not record anything when requested to.
        m_FrameSnapshot = std::make_unique<NullFrameSnapshot>(); 

        m_ReSTIR = std::make_unique<ReSTIR>();
        m_ReSTIR->Initialize(rSettings);
    }

    void WaveFrontRenderer::BeginSnapshot()
    {
        // Replacing the snapshot with a non-null one will start recording requested features.
        m_FrameSnapshot = std::make_unique<FrameSnapshot>();
    }

    std::unique_ptr<FrameSnapshot> WaveFrontRenderer::EndSnapshot()
    {
        // Move the snapshot to a temporary variable to return shortly
        auto snap = std::move(m_FrameSnapshot);
        // Make the snapshot a Null once again to stop recording
        m_FrameSnapshot = std::make_unique<NullFrameSnapshot>();

        return std::move(snap);
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
        }
        return std::make_unique<MemoryBuffer>(vertices);
    }

    std::unique_ptr<Lumen::ILumenPrimitive> WaveFrontRenderer::CreatePrimitive(PrimitiveData& a_PrimitiveData)
    {
        //TODO let optix build the acceleration structure and return the handle.

        auto vertexBuffer = InterleaveVertexData(a_PrimitiveData);
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
        
        //printf("Index buffer Size %i \n", static_cast<int>(correctedIndices.size()));
        std::unique_ptr<MemoryBuffer> indexBuffer = std::make_unique<MemoryBuffer>(correctedIndices);

        uint8_t numVertices = sizeof(vertexBuffer) / sizeof(Vertex);
        //vertexBuffer->GetDevicePtr<Vertex>();
        std::unique_ptr<MemoryBuffer> emissiveBuffer = std::make_unique<MemoryBuffer>((numVertices / 3) * sizeof(bool));
        //std::unique_ptr<MemoryBuffer> indexBuffer = std::make_unique<MemoryBuffer>(a_PrimitiveData.m_IndexBinary);

        //std::unique_ptr<MemoryBuffer> primMat = std::make_unique<MemoryBuffer>(a_PrimitiveData.m_Material);
        //std::unique_ptr<MemoryBuffer> primMat = static_cast<Material*>(a_PrimitiveData.m_Material.get())->GetDeviceMaterial();

        FindEmissives(vertexBuffer->GetDevicePtr<Vertex>(), emissiveBuffer->GetDevicePtr<bool>(), indexBuffer->GetDevicePtr<uint32_t>(), static_cast<Material*>(a_PrimitiveData.m_Material.get())->GetDeviceMaterial(), numVertices);

        // add bool buffer pointer to device prim pointer


        


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
        buildInput.triangleArray.numVertices = a_PrimitiveData.m_Positions.Size();
        buildInput.triangleArray.numSbtRecords = 1;
        buildInput.triangleArray.flags = &geomFlags;

        auto gAccel = m_OptixSystem->BuildGeometryAccelerationStructure(buildOptions, buildInput);

        auto prim = std::make_unique<PTPrimitive>(std::move(vertexBuffer), std::move(indexBuffer), std::move(emissiveBuffer), std::move(gAccel));

        prim->m_Material = a_PrimitiveData.m_Material;

        prim->m_SceneDataTableEntry = m_Table->AddEntry<DevicePrimitive>();
        auto& entry = prim->m_SceneDataTableEntry.GetData();
        entry.m_VertexBuffer = prim->m_VertBuffer->GetDevicePtr<Vertex>();
        entry.m_IndexBuffer = prim->m_IndexBuffer->GetDevicePtr<unsigned int>();
        entry.m_Material = static_cast<Material*>(prim->m_Material.get())->GetDeviceMaterial();
        entry.m_IsEmissive = prim->m_BoolBuffer->GetDevicePtr<bool>();

        return prim;
    }

    std::shared_ptr<Lumen::ILumenMesh> WaveFrontRenderer::CreateMesh(
        std::vector<std::unique_ptr<Lumen::ILumenPrimitive>>& a_Primitives)
    {
        //TODO Let optix build the medium level acceleration structure and return the mesh handle for it.

        auto mesh = std::make_shared<PTMesh>(a_Primitives, m_ServiceLocator);
        return mesh;
    }

    std::shared_ptr<Lumen::ILumenTexture> WaveFrontRenderer::CreateTexture(void* a_PixelData,
        uint32_t a_Width, uint32_t a_Height)
    {
        static cudaChannelFormatDesc formatDesc = cudaCreateChannelDesc<uchar4>();
        return std::make_shared<Texture>(a_PixelData, formatDesc, a_Width, a_Height);
    }

    std::shared_ptr<Lumen::ILumenMaterial> WaveFrontRenderer::CreateMaterial(
        const MaterialData& a_MaterialData)
    {
        auto mat = std::make_shared<Material>();
        mat->SetDiffuseColor(a_MaterialData.m_DiffuseColor);
        mat->SetDiffuseTexture(a_MaterialData.m_DiffuseTexture);
        mat->SetEmission(a_MaterialData.m_EmssivionVal);
        return mat;
    }

    std::shared_ptr<Lumen::ILumenVolume> WaveFrontRenderer::CreateVolume(const std::string& a_FilePath)
    {
        //TODO tell optix to create a volume acceleration structure.
        std::shared_ptr<Lumen::ILumenVolume> volume = std::make_shared<PTVolume>(a_FilePath, m_ServiceLocator);

        //volumetric_bookmark
    //TODO: add volume records to sbt
    /*volume->m_RecordHandle = m_ShaderBindingTableGenerator->AddHitGroup<DeviceVolume>();
    auto& rec = volume->m_RecordHandle.GetRecord();
    rec.m_Header = GetProgramGroupHeader("VolumetricHit");
    rec.m_Data.m_Grid = volume->m_Handle.grid<float>();*/

        uint32_t geomFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE };

        OptixAccelBuildOptions buildOptions = {};
        buildOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        buildOptions.motionOptions = {};

        OptixAabb aabb = { -1.5f, -1.5f, -1.5f, 1.5f, 1.5f, 1.5f };

        auto grid = std::static_pointer_cast<PTVolume>(volume)->GetHandle()->grid<float>();
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

        std::static_pointer_cast<PTVolume>(volume)->m_AccelerationStructure = m_OptixSystem->BuildGeometryAccelerationStructure(buildOptions, buildInput);

        return volume;
    }

    std::shared_ptr<Lumen::ILumenScene> WaveFrontRenderer::CreateScene(SceneData a_SceneData)
    {
        return std::make_shared<PTScene>(a_SceneData, m_ServiceLocator);
    }

    WaveFrontRenderer::WaveFrontRenderer() : m_FrameIndex(0), m_CUDAContext(nullptr)
    {

    }

    unsigned WaveFrontRenderer::TraceFrame(std::shared_ptr<Lumen::ILumenScene>& a_Scene)
    {
        //add to lights buffer? in traceframe

        //add lights to mesh in scene
            //add mesh
            //keep instances in scene
            //for each instance, add all emissive triangles to light buffer with world space pos

        //Get instances from scene
            //check which instances are emissives to optimize the looping over instances
            //inside of these instances you compare which triangles are emissive through boolean buffer
            //add those to lights buffer in world space
                //where to keep lights buffer?? - scene! yes!

        //Index of the current and last frame to access buffers.
        const auto currentIndex = m_FrameIndex;
        const auto temporalIndex = m_FrameIndex == 1 ? 0 : 1;

        //Data needed in the algorithms.
        const uint32_t numPixels = m_Settings.renderResolution.x * m_Settings.renderResolution.y;

        //Start by clearing the data from the previous frame.
        ResetLightChannels(m_PixelBufferSeparate.GetDevicePtr<float3>(), numPixels, static_cast<unsigned>(LightChannel::NUM_CHANNELS));
        ResetLightChannels(m_PixelBufferCombined.GetDevicePtr<float3>(), numPixels, 1);
        cudaDeviceSynchronize();
        CHECKLASTCUDAERROR;

        //Generate camera rays.
        glm::vec3 eye, u, v, w;
        a_Scene->m_Camera->SetAspectRatio(static_cast<float>(m_Settings.renderResolution.x) / static_cast<float>(m_Settings.renderResolution.y));
        a_Scene->m_Camera->GetVectorData(eye, u, v, w);

        //Camera forward direction.
        const float3 camForward = { w.x, w.y, w.z };
        const float3 camPosition = { eye.x, eye.y, eye.z };

        float3 eyeCuda, uCuda, vCuda, wCuda;
        eyeCuda = make_float3(eye.x, eye.y, eye.z);
        uCuda = make_float3(u.x, u.y, u.z);
        vCuda = make_float3(v.x, v.y, v.z);
        wCuda = make_float3(w.x, w.y, w.z);

        printf("Camera pos: %f %f %f\n", camPosition.x, camPosition.y, camPosition.z);

        //Increment framecount each frame.
        static unsigned frameCount = 0;
        ++frameCount;

        const WaveFront::PrimRayGenLaunchParameters::DeviceCameraData cameraData(eyeCuda, uCuda, vCuda, wCuda);
        auto rayPtr = m_Rays.GetDevicePtr<AtomicBuffer<IntersectionRayData>>();
        const PrimRayGenLaunchParameters primaryRayGenParams(
            uint2{m_Settings.renderResolution.x, m_Settings.renderResolution.y}, 
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
            resBuffers["Origins"].m_Memory = std::make_unique<CudaGLTexture>(GL_RGB32F, m_Settings.renderResolution.x,
                m_Settings.renderResolution.y, 3 * sizeof(float));

            resBuffers["Directions"].m_Memory = std::make_unique<CudaGLTexture>(GL_RGB32F, m_Settings.renderResolution.x,
                m_Settings.renderResolution.y, 3 * sizeof(float));

            resBuffers["Contributions"].m_Memory = std::make_unique<CudaGLTexture>(GL_RGB32F ,m_Settings.renderResolution.x,
                m_Settings.renderResolution.y, 3 * sizeof(float));

            // A CUDA kernel used to separate the interleave primary ray buffer into 3 different buffers
            // This is the main reason we use a lambda, as it needs to be defined how to interpret the data
            SeparateIntersectionRayBufferCPU((m_Rays.GetSize() - sizeof(uint32_t)) / sizeof(IntersectionRayData),
                m_Rays.GetDevicePtr<AtomicBuffer<IntersectionRayData>>(),
                resBuffers.at("Origins").m_Memory->GetDevicePtr<float3>(),
                resBuffers.at("Directions").m_Memory->GetDevicePtr<float3>(),
                resBuffers.at("Contributions").m_Memory->GetDevicePtr<float3>());

            return resBuffers;
        });

        m_FrameSnapshot->AddBuffer([&]()
        {
            auto motionVectorBuffer = m_MotionVectors.GetMotionVectorBuffer();
        	
            std::map<std::string, FrameSnapshot::ImageBuffer> resBuffers;
            resBuffers["Motion vector direction"].m_Memory = std::make_unique<CudaGLTexture>(GL_RGB32F, m_Settings.renderResolution.x,
                m_Settings.renderResolution.y, 3 * sizeof(float));

            resBuffers["Motion vector magnitude"].m_Memory = std::make_unique<CudaGLTexture>(GL_RGB32F, m_Settings.renderResolution.x,
                m_Settings.renderResolution.y, 3 * sizeof(float));

           SeparateMotionVectorBufferCPU(m_Settings.renderResolution.x * m_Settings.renderResolution.y,
               motionVectorBuffer->GetDevicePtr<MotionVectorBuffer>(),
                resBuffers.at("Motion vector direction").m_Memory->GetDevicePtr<float3>(),
                resBuffers.at("Motion vector magnitude").m_Memory->GetDevicePtr<float3>()
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

        //Retrieve the acceleration structure and scene data table once.
        m_OptixSystem->UpdateSBT();
        auto* sceneDataTableAccessor = m_Table->GetDevicePointer();
        auto accelerationStructure = std::static_pointer_cast<PTScene>(a_Scene)->GetSceneAccelerationStructure();
        cudaDeviceSynchronize();
        CHECKLASTCUDAERROR;

        //Pass the buffers to the optix shader for shading.
        OptixLaunchParameters rayLaunchParameters;
        rayLaunchParameters.m_TraceType = RayType::INTERSECTION_RAY;
        rayLaunchParameters.m_MinMaxDistance = { 0.01f, 5000.f };
        rayLaunchParameters.m_IntersectionBuffer = m_IntersectionData.GetDevicePtr<AtomicBuffer<IntersectionData>>();
        rayLaunchParameters.m_IntersectionRayBatch = m_Rays.GetDevicePtr<AtomicBuffer<IntersectionRayData>>();
        rayLaunchParameters.m_TraversableHandle = accelerationStructure;
        rayLaunchParameters.m_ResolutionAndDepth = uint3{ m_Settings.renderResolution.x, m_Settings.renderResolution.y, m_Settings.depth };

        //The settings for shadow ray resolving.
        OptixLaunchParameters shadowRayLaunchParameters;
        shadowRayLaunchParameters = rayLaunchParameters;
        shadowRayLaunchParameters.m_ResultBuffer = m_PixelBufferSeparate.GetDevicePtr<float3>();
        shadowRayLaunchParameters.m_ShadowRayBatch = m_ShadowRays.GetDevicePtr<AtomicBuffer<ShadowRayData>>();
        shadowRayLaunchParameters.m_TraceType = RayType::SHADOW_RAY;

        //Set the amount of rays to trace. Initially same as screen size.
        auto numIntersectionRays = numPixels;

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

            const auto surfaceDataBufferIndex = depth == 0 ? currentIndex : 2;   //1 and 2 are used for the first intersection and remembered for temporal use.
            ExtractSurfaceData(
                numIntersections,
                m_IntersectionData.GetDevicePtr<AtomicBuffer<IntersectionData>>(),
                m_Rays.GetDevicePtr<AtomicBuffer<IntersectionRayData>>(),
                m_SurfaceData[surfaceDataBufferIndex].GetDevicePtr<SurfaceData>(),
                sceneDataTableAccessor);
            cudaDeviceSynchronize();
            CHECKLASTCUDAERROR;

            //motion vector generation
            if (depth == 0)
            {
                glm::mat4 previousFrameMatrix, currentFrameMatrix;
                a_Scene->m_Camera->GetMatrixData(previousFrameMatrix, currentFrameMatrix);
                sutil::Matrix4x4 prevFrameMatrixArg = ConvertGLMtoSutilMat4(previousFrameMatrix);

                glm::mat4 projectionMatrix = a_Scene->m_Camera->GetProjectionMatrix();
                sutil::Matrix4x4 projectionMatrixArg = ConvertGLMtoSutilMat4(projectionMatrix);

                MotionVectorsGenerationData motionVectorsGenerationData;
                motionVectorsGenerationData.m_MotionVectorBuffer = nullptr;
                motionVectorsGenerationData.a_CurrentSurfaceData = m_SurfaceData[currentIndex].GetDevicePtr<SurfaceData>();
                motionVectorsGenerationData.m_ScreenResolution = make_uint2(m_Settings.renderResolution.x, m_Settings.renderResolution.y);
                motionVectorsGenerationData.m_PrevViewMatrix = prevFrameMatrixArg.inverse();
                motionVectorsGenerationData.m_ProjectionMatrix = projectionMatrixArg;
                m_MotionVectors.Update(motionVectorsGenerationData);
            }

            //TODO add ReSTIR instance and run from shading kernel.

            /*
             * Call the shading kernels.
             */
            ShadingLaunchParameters shadingLaunchParams(
                uint3{ m_Settings.renderResolution.x, m_Settings.renderResolution.y, m_Settings.depth },
                m_SurfaceData[currentIndex].GetDevicePtr<SurfaceData>(),
                m_SurfaceData[temporalIndex].GetDevicePtr<SurfaceData>(),
                m_ShadowRays.GetDevicePtr<AtomicBuffer<ShadowRayData>>(),
                m_TriangleLights.GetDevicePtr<TriangleLight>(),
                3,
                camPosition,
                camForward,
                accelerationStructure,  //ReSTIR needs to check visibility early on so it does an optix launch for this scene.
                m_OptixSystem.get(),
                WangHash(frameCount), //Use the frame count as the random seed.
                m_ReSTIR.get(), //ReSTIR, can not be nullptr.
                depth,  //The current depth.
                m_MotionVectors.GetMotionVectorBuffer()->GetDevicePtr<MotionVectorBuffer>(),
                m_PixelBufferSeparate.GetDevicePtr<float3>()
            );

            //Reset the atomic counter.
            ResetAtomicBuffer<IntersectionRayData>(&m_Rays);
            cudaDeviceSynchronize();

            Shade(shadingLaunchParams);
            cudaDeviceSynchronize();
            CHECKLASTCUDAERROR;

            //Set the number of intersection rays to the size of the ray buffer.
            m_Rays.Read(&numIntersectionRays, sizeof(uint32_t), 0);

            //Reset the atomic counters for the next wave. Also clear the surface data at depth 2 (the one that is overwritten each wave).
            cudaMemset(m_SurfaceData[2].GetDevicePtr(), 0, sizeof(SurfaceData) * numPixels);
            cudaDeviceSynchronize();
            CHECKLASTCUDAERROR;

            m_IntersectionData.Write(counterDefault);

            //Swap the ReSTIR buffers around.
            m_ReSTIR->SwapBuffers();
        }
    	
        //The amount of shadow rays to trace.
        unsigned numShadowRays = 0;
        m_ShadowRays.Read(&numShadowRays, sizeof(uint32_t), 0);

        if(numShadowRays > 0)
        {
            //Tell optix to resolve the shadow rays.
            m_OptixSystem->TraceRays(numShadowRays, shadowRayLaunchParameters);
            cudaDeviceSynchronize();
            CHECKLASTCUDAERROR;
        }

        PostProcessLaunchParameters postProcessLaunchParams(
            m_Settings.renderResolution,
            m_Settings.outputResolution,
            m_PixelBufferSeparate.GetDevicePtr<float3>(),
            m_PixelBufferCombined.GetDevicePtr<float3>(),
            m_OutputBuffer.GetDevicePtr()
        );

        //Post processing using CUDA kernel.
        PostProcess(postProcessLaunchParams);
        cudaDeviceSynchronize();
        CHECKLASTCUDAERROR;
        
        //Change frame index 0..1
        ++m_FrameIndex;
        if(m_FrameIndex == 2)
        {
            m_FrameIndex = 0;
        }

        a_Scene->m_Camera->UpdatePreviousFrameMatrix();
        ++frameCount;

        m_DebugTexture = m_OutputBuffer.GetTexture();
//#if defined(_DEBUG)
        m_MotionVectors.GenerateDebugTextures();
        //m_DebugTexture = m_MotionVectors.GetMotionVectorMagnitudeTex();
        m_DebugTexture = m_MotionVectors.GetMotionVectorDirectionsTex();
//#endif
    	
        //Return the GLuint texture ID.
        return m_OutputBuffer.GetTexture();
    }
}
#endif