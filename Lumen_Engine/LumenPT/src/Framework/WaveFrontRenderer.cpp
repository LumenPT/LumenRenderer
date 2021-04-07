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
#include "OutputBuffer.h"
#include "SceneDataTable.h"
#include "../CUDAKernels/WaveFrontKernels.cuh"
#include "../Shaders/CppCommon/LumenPTConsts.h"
#include "../Shaders/CppCommon/WaveFrontDataStructs.h"
#include "CudaUtilities.h"

#include <Optix/optix_function_table_definition.h>
#include <filesystem>
#include <glm/gtx/compatibility.hpp>

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
        optixInitData.m_SolidProgramData.m_ProgramPath = LumenPTConsts::gs_ShaderPathBase + "WaveFrontShaders.ptx";;
        optixInitData.m_SolidProgramData.m_ProgramLaunchParamName = "launchParams";
        optixInitData.m_SolidProgramData.m_ProgramRayGenFuncName = "__raygen__WaveFrontRG";
        optixInitData.m_SolidProgramData.m_ProgramMissFuncName = "__miss__WaveFrontMS";
        optixInitData.m_SolidProgramData.m_ProgramAnyHitFuncName = "__anyhit__WaveFrontAH";
        optixInitData.m_SolidProgramData.m_ProgramClosestHitFuncName = "__closesthit__WaveFrontCH";
        optixInitData.m_VolumetricProgramData.m_ProgramPath = LumenPTConsts::gs_ShaderPathBase + "volumetric_wavefront.ptx";
        optixInitData.m_VolumetricProgramData.m_ProgramIntersectionFuncName = "__intersection__Volumetric";
        optixInitData.m_VolumetricProgramData.m_ProgramAnyHitFuncName = "__anyhit__Volumetric";
        optixInitData.m_VolumetricProgramData.m_ProgramClosestHitFuncName = "__closesthit__Volumetric";
        optixInitData.m_PipelineMaxNumHitResultAttributes = 2;
        optixInitData.m_PipelineMaxNumPayloads = 2;

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
        const auto numShadowRays = numPixels * m_Settings.depth; //TODO: change to 2x num pixels and add safety check to resolve when full.
        m_Rays.Resize(sizeof(AtomicBuffer<IntersectionRayData>) + sizeof(IntersectionRayData) * numPrimaryRays);
        m_ShadowRays.Resize(sizeof(AtomicBuffer<ShadowRayData>) + sizeof(ShadowRayData) * numShadowRays);
        m_VolumetricShadowRays.Resize(sizeof(AtomicBuffer<ShadowRayData>) + sizeof(ShadowRayData) * m_Settings.depth);

        //Reset Atomic Counters for Intersection and Shadow Rays
        m_Rays.Write(0);
        m_ShadowRays.Write(0);
        m_VolumetricShadowRays.Write(0);

        //Initialize the intersection data. This one is the size of numPixels maximum.
		//TODO: increase size of volume intersection buffer to allow multiple consecutive volumes
        m_IntersectionData.Resize(sizeof(AtomicBuffer<IntersectionData>) + sizeof(IntersectionData) * numPixels);
		m_VolumetricIntersectionData.Resize(sizeof(AtomicBuffer <IntersectionData>) + sizeof(VolumetricIntersectionData) * numPixels);

        //Reset Atomic Counter for Intersection Buffers.
        m_IntersectionData.Write(0);
		m_VolumetricIntersectionData.Write(0);

        //Initialize each surface data buffer.
        for(auto& surfaceDataBuffer : m_SurfaceData)
        {
            //Note; Only allocates memory and stores the size on the GPU. It does not actually fill any data in yet.
            surfaceDataBuffer.Resize(numPixels * sizeof(SurfaceData));
        }

        for(auto& volumetricDataBuffer : m_VolumetricData)
        {
            volumetricDataBuffer.Resize(numPixels * sizeof(VolumetricData));
        }

        //TODO: number of lights will be dynamic per frame but this is temporary.
        constexpr auto numLights = 3;

        m_TriangleLights.Resize(sizeof(TriangleLight) * numLights);

        //Temporary lights, stored in the buffer.
        float3 pos1, pos2, pos3;
        pos1 = { 150.f, 41.f, 210.f };
        pos2 = { 150.f, 41.f, 110.f };
        pos3 = { 90.f, 41.f, 210.f };

        float i1 = 3000000.f;
        float i2 = 300000.f;
        float i3 = 699999.f;

        TriangleLight lights[numLights] =
        {
            {pos1, pos1, pos1, {0.f, 0.f, 0.f}, {i1, i1, i1}, 10.f},
            {pos2, pos2, pos2, {0.f, 0.f, 0.f}, {i2, i2, i2}, 10.f},
            {pos3, pos3, pos3, {0.f, 0.f, 0.f}, {i3, i3, i3}, 10.f}
        };

        m_TriangleLights.Write(&lights[0], sizeof(TriangleLight) * numLights, 0);

        //Set the service locator pointer to point to the m'table.
        m_Table = std::make_unique<SceneDataTable>();
        m_ServiceLocator.m_SceneDataTable = m_Table.get();

        m_ServiceLocator.m_Renderer = this;

        

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

        auto prim = std::make_unique<PTPrimitive>(std::move(vertexBuffer), std::move(indexBuffer), std::move(gAccel));

        prim->m_Material = a_PrimitiveData.m_Material;

        prim->m_SceneDataTableEntry = m_Table->AddEntry<DevicePrimitive>();
        auto& entry = prim->m_SceneDataTableEntry.GetData();
        entry.m_VertexBuffer = prim->m_VertBuffer->GetDevicePtr<Vertex>();
        entry.m_IndexBuffer = prim->m_IndexBuffer->GetDevicePtr<unsigned int>();
        entry.m_Material = std::static_pointer_cast<Material>(prim->m_Material)->GetDeviceMaterial();

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

        volume->m_SceneDataTableEntry = m_Table->AddEntry<DeviceVolume>();
        auto& entry = volume->m_SceneDataTableEntry.GetData();
        entry.m_Grid = grid;

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

        float3 eyeCuda, uCuda, vCuda, wCuda;
        eyeCuda = make_float3(eye.x, eye.y, eye.z);
        uCuda = make_float3(u.x, u.y, u.z);
        vCuda = make_float3(v.x, v.y, v.z);
        wCuda = make_float3(w.x, w.y, w.z);

        static unsigned frameCount = 0;

        const WaveFront::PrimRayGenLaunchParameters::DeviceCameraData cameraData(eyeCuda, uCuda, vCuda, wCuda);
        auto rayPtr = m_Rays.GetDevicePtr<AtomicBuffer<IntersectionRayData>>();
        const PrimRayGenLaunchParameters primaryRayGenParams(
            uint2{m_Settings.renderResolution.x, m_Settings.renderResolution.y}, 
            cameraData, 
            rayPtr, frameCount);   //TODO what is framecount?
        GeneratePrimaryRays(primaryRayGenParams);
        cudaDeviceSynchronize();
        CHECKLASTCUDAERROR;

        m_Rays.Write(numPixels, 0); //Set the counter to be equal to the amount of rays being shot. This is manual because the atomic is not used yet.

        //Clear the surface data that contains information from the second last frame so that it can be reused by this frame.
        cudaMemset(m_SurfaceData[currentIndex].GetDevicePtr(), 0, sizeof(SurfaceData) * numPixels);
        cudaDeviceSynchronize();
        CHECKLASTCUDAERROR;

        //Set the shadow ray count to 0.
        const unsigned counterDefault = 0;
        m_ShadowRays.Write(counterDefault);
        m_VolumetricShadowRays.Write(counterDefault);
        m_IntersectionData.Write(counterDefault);
		m_VolumetricIntersectionData.Write(counterDefault);

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
		rayLaunchParameters.m_VolumetricIntersectionBuffer = m_VolumetricIntersectionData.GetDevicePtr<AtomicBuffer<VolumetricIntersectionData>>();
        rayLaunchParameters.m_IntersectionRayBatch = m_Rays.GetDevicePtr<AtomicBuffer<IntersectionRayData>>();
        rayLaunchParameters.m_SceneData = sceneDataTableAccessor;
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
        for(unsigned depth = 0; depth < m_Settings.depth && numIntersectionRays > 0; ++depth)
        {
            //Tell Optix to resolve the primary rays that have been generated.
            m_OptixSystem->TraceRays(numIntersectionRays, rayLaunchParameters);
            cudaDeviceSynchronize();
            CHECKLASTCUDAERROR;

            /*
             * Calculate the surface data for this depth.
             */
            unsigned numIntersections = 0;
            m_IntersectionData.Read(&numIntersections, sizeof(numIntersections), 0);
            const auto surfaceDataBufferIndex = depth == 0 ? currentIndex : 2;   //1 and 2 are used for the first intersection and remembered for temporal use.
            ExtractSurfaceData(
                numIntersections, 
                m_IntersectionData.GetDevicePtr<AtomicBuffer<IntersectionData>>(), 
                m_Rays.GetDevicePtr<AtomicBuffer<IntersectionRayData>>(), 
                m_SurfaceData[surfaceDataBufferIndex].GetDevicePtr<SurfaceData>(), 
                sceneDataTableAccessor);

            unsigned numVolumeIntersections = 0;
            m_VolumetricIntersectionData.Read(&numVolumeIntersections, sizeof(numVolumeIntersections), 0);
            const auto volumetricDataBufferIndex = 0;
            ExtractVolumetricData(
                numVolumeIntersections,
                m_VolumetricIntersectionData.GetDevicePtr<AtomicBuffer<VolumetricIntersectionData>>(),
                m_Rays.GetDevicePtr<AtomicBuffer<IntersectionRayData>>(),
                m_VolumetricData[volumetricDataBufferIndex].GetDevicePtr<VolumetricData>(),
                sceneDataTableAccessor);

            cudaDeviceSynchronize();
            CHECKLASTCUDAERROR;

            //TODO add ReSTIR instance and run from shading kernel.

            /*
             * Call the shading kernels.
             */
            ShadingLaunchParameters shadingLaunchParams(
                uint3{m_Settings.renderResolution.x, m_Settings.renderResolution.y, m_Settings.depth},
                m_SurfaceData[currentIndex].GetDevicePtr<SurfaceData>(),
                m_SurfaceData[temporalIndex].GetDevicePtr<SurfaceData>(),
                m_VolumetricData[volumetricDataBufferIndex].GetDevicePtr<VolumetricData>(), //TODO
                m_ShadowRays.GetDevicePtr<AtomicBuffer<ShadowRayData>>(),
                m_VolumetricShadowRays.GetDevicePtr<AtomicBuffer<ShadowRayData>>(),
                m_TriangleLights.GetDevicePtr<TriangleLight>(),
                m_TriangleLights.GetSize() / sizeof(TriangleLight),  //TODO hard coded for now but will be updated dynamically.
                nullptr,    //TODO get CDF from ReSTIR.
                m_PixelBufferSeparate.GetDevicePtr<float3>()
            );

            //Reset the atomic counter.
            m_Rays.Write(counterDefault);
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
            m_VolumetricIntersectionData.Write(counterDefault);
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
            m_OutputBuffer.GetDevicePointer()
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

        ++frameCount;

        //Return the GLuint texture ID.
        return m_OutputBuffer.GetTexture();
    }
}
#endif