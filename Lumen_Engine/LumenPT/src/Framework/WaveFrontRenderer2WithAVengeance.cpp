#if defined(WAVEFRONT)

#include "../Shaders/CppCommon/WaveFrontDataStructs.h"
#include "WaveFrontRenderer2WithAVengeance.h"
#include "WaveFrontRenderer.h"
#include "PTMesh.h"
#include "PTScene.h"
#include "Material.h"
#include "Texture.h"
#include "PTVolume.h"
#include "Material.h"
#include "MemoryBuffer.h"
#include "OutputBuffer.h"
#include "../CUDAKernels/WaveFrontKernels.cuh"
#include <filesystem>
#include <glm/gtx/compatibility.hpp>

namespace WaveFront
{
    void WaveFrontRenderer2WithAVengeance::Init(const WaveFrontSettings& a_Settings)
    {
        m_FrameIndex = 0;
        m_Settings = a_Settings;

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

        //Initialize the intersection data. This one is the size of numPixels maximum.
        m_IntersectionData.Resize(sizeof(AtomicBuffer<IntersectionData>) + sizeof(IntersectionData) * numPixels);

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
        TriangleLight lights[numLights] = {
            {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}},
            {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}},
            {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}} };
        m_TriangleLights.Write(&lights[0], sizeof(TriangleLight) * numLights, 0);

        //Set up material buffer. Initial space for 1024 meshes.
        m_Materials = std::make_unique<MemoryBuffer>(1024 * sizeof(void*));

        //TODO: initialize optix system.
        //TODO: I don't think wavefront should care about this data at all. 
        OptixWrapper::InitializationData optixInitData;

        m_OptixSystem = std::make_unique<OptixWrapper>(optixInitData);
    }

    std::unique_ptr<Lumen::ILumenPrimitive> WaveFrontRenderer2WithAVengeance::CreatePrimitive(PrimitiveData& a_MeshData)
    {
        //TODO let optix build the acceleration structure and return the handle.
    }

    std::shared_ptr<Lumen::ILumenMesh> WaveFrontRenderer2WithAVengeance::CreateMesh(
        std::vector<std::unique_ptr<Lumen::ILumenPrimitive>>& a_Primitives)
    {
        //TODO Let optix build the medium level acceleration structure and return the mesh handle for it.
    }

    std::shared_ptr<Lumen::ILumenTexture> WaveFrontRenderer2WithAVengeance::CreateTexture(void* a_PixelData,
        uint32_t a_Width, uint32_t a_Height)
    {
        static cudaChannelFormatDesc formatDesc = cudaCreateChannelDesc<uchar4>();
        return std::make_shared<Texture>(a_PixelData, formatDesc, a_Width, a_Height);
    }

    std::shared_ptr<Lumen::ILumenMaterial> WaveFrontRenderer2WithAVengeance::CreateMaterial(
        const MaterialData& a_MaterialData)
    {
        auto mat = std::make_shared<Material>();
        mat->SetDiffuseColor(a_MaterialData.m_DiffuseColor);
        mat->SetDiffuseTexture(a_MaterialData.m_DiffuseTexture);
        mat->SetEmission(a_MaterialData.m_EmssivionVal);
        return mat;
    }

    std::shared_ptr<Lumen::ILumenVolume> WaveFrontRenderer2WithAVengeance::CreateVolume(const std::string& a_FilePath)
    {
        //TODO tell optix to create a volume acceleration structure.
    }

    WaveFrontRenderer2WithAVengeance::WaveFrontRenderer2WithAVengeance() : m_FrameIndex(0)
    {

    }

    unsigned WaveFrontRenderer2WithAVengeance::TraceFrame()
    {
        //Index of the current and last frame to access buffers.
        const auto currentIndex = m_FrameIndex;
        const auto temporalIndex = m_FrameIndex == 1 ? 0 : 1;

        //Data needed in the algorithms.
        const unsigned numPixels = m_Settings.renderResolution.x * m_Settings.renderResolution.y;

        //Start by clearing the data from the previous frame.
        ResetLightChannels(m_PixelBufferCombined.GetDevicePtr<float3>(), numPixels, static_cast<unsigned>(LightChannel::NUM_CHANNELS));
        ResetLightChannels(m_PixelBufferSeparate.GetDevicePtr<float3>(), numPixels, 1);

        //Generate camera rays.
        float3 eye, u, v, w;
        m_Camera.SetAspectRatio(static_cast<float>(m_Settings.renderResolution.x) / static_cast<float>(m_Settings.renderResolution.y));
        m_Camera.GetVectorData(eye, u, v, w);
        const WaveFront::PrimRayGenLaunchParameters::DeviceCameraData cameraData(eye, u, v, w);
        auto rayPtr = m_Rays.GetDevicePtr<AtomicBuffer<IntersectionRayData>>();
        const PrimRayGenLaunchParameters primaryRayGenParams(uint2{m_Settings.renderResolution.x, m_Settings.renderResolution.y}, cameraData, rayPtr, 1);   //TODO what is framecount?
        GeneratePrimaryRays(primaryRayGenParams);
        m_Rays.Write(numPixels, 0); //Set the counter to be equal to the amount of rays being shot. This is manual because the atomic is not used yet.

        //Clear the surface data that contains information from the second last frame so that it can be reused by this frame.
        cudaMemset(m_SurfaceData[currentIndex].GetDevicePtr(), 0, sizeof(SurfaceData) * numPixels);

        //Set the shadow ray count to 0.
        const unsigned counterDefault = 0;
        m_ShadowRays.Write(&counterDefault, sizeof(unsigned), 0);

        //Pass the buffers to the optix shader for shading.
        OptixLaunchParameters rayLaunchParameters;
        rayLaunchParameters.m_ResolutionAndDepth = uint3{ m_Settings.renderResolution.x, m_Settings.renderResolution.y, m_Settings.depth };
        const float2 minMaxDistanceRay = { 0.005f, 99999999.f };
        //TODO set the params (pass the buffers). Atomic so gotta change signature.

        //The settings for shadow ray resolving.
        OptixLaunchParameters shadowRayLaunchParameters;

        /*
         * Resolve rays and shade at every depth.
         */
        for(unsigned depth = 0; depth < m_Settings.depth; ++depth)
        {
            //Tell Optix to resolve the primary rays that have been generated.
            m_OptixSystem->TraceRays(RayType::INTERSECTION_RAY, rayLaunchParameters, minMaxDistanceRay);

            /*
             * Calculate the surface data for this depth.
             */
            //TODO: Replace material pointer with actual pointer.
            unsigned numIntersections = 0;
            m_IntersectionData.Read(&numIntersections, sizeof(unsigned), 0);
            const auto surfaceDataBufferIndex = depth == 0 ? currentIndex : 2;   //1 and 2 are used for the first intersection and remembered for temporal use.
            ExtractSurfaceData(numIntersections, m_IntersectionData.GetDevicePtr<AtomicBuffer<IntersectionData>>(), m_Rays.GetDevicePtr<AtomicBuffer<IntersectionRayData>>(), m_SurfaceData[surfaceDataBufferIndex].GetDevicePtr<SurfaceData>(), nullptr);

            //TODO add ReSTIR instance and run from shading kernel.

            /*
             * Call the shading kernels.
             */
            ShadingLaunchParameters shadingLaunchParams(
                uint3{m_Settings.renderResolution.x, m_Settings.renderResolution.y, m_Settings.depth},
                m_SurfaceData[currentIndex].GetDevicePtr<SurfaceData>(),
                m_SurfaceData[temporalIndex].GetDevicePtr<SurfaceData>(),
                m_ShadowRays.GetDevicePtr<AtomicBuffer<ShadowRayData>>(),
                m_TriangleLights.GetDevicePtr<TriangleLight>(),
                3,  //TODO hard coded for now but will be updated dynamically.
                nullptr,    //TODO get CDF from ReSTIR.
                m_PixelBufferSeparate.GetDevicePtr<float3>()
            );

            Shade(shadingLaunchParams);

            //Reset the atomic counters for the next wave. Also clear the surface data at depth 2 (the one that is overwritten each wave).
            cudaMemset(m_SurfaceData[2].GetDevicePtr(), 0, sizeof(SurfaceData) * numPixels);
            m_IntersectionData.Write(&counterDefault, sizeof(unsigned), 0);
            m_Rays.Write(&counterDefault, sizeof(unsigned), 0);
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
        
        //Change frame index 0..1
        ++m_FrameIndex;
        if(m_FrameIndex == 2)
        {
            m_FrameIndex = 0;
        }

        //Return the GLuint texture ID.
        return m_OutputBuffer.GetTexture();
    }

    void WaveFrontRenderer2WithAVengeance::SetInstanceMaterial(unsigned a_InstanceId,
        std::shared_ptr<Lumen::ILumenMaterial>& a_Material)
    {
        //Current size in pointers.
        const auto size = m_Materials->GetSize() / sizeof(void*);

        //Allocate more memory if not enough is available.
        if(size <= a_InstanceId)
        {
            std::unique_ptr<MemoryBuffer> temp = std::make_unique<MemoryBuffer>(m_Materials->GetSize() * 2);
            temp->CopyFrom(*m_Materials, m_Materials->GetSize(), 0, 0);
            m_Materials = std::move(temp);
        }

        auto asMaterial = std::static_pointer_cast<Material>(a_Material);

        //Set the device material pointer for the ID.
        m_Materials->Write(asMaterial->GetDeviceMaterial(), sizeof(void*), a_InstanceId * sizeof(void*));
    }
}
#endif