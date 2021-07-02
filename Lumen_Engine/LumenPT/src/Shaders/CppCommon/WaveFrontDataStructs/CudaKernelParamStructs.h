#pragma once
#include "../WaveFrontDataStructs.h"
#include "../CudaDefines.h"
#include "SurfaceData.h"
#include "VolumetricData.h"
#include "MotionVectorsGenerationData.h"
#include <array>

class ReSTIR;
struct FrameStats;
namespace WaveFront
{
    class OptixWrapper;

    //Kernel Launch parameters
    struct PrimRayGenLaunchParameters
    {
        //Camera data
        struct DeviceCameraData
        {

            CPU_ONLY DeviceCameraData(
                const float3& a_Position,
                const float3& a_Up,
                const float3& a_Right,
                const float3& a_Forward)
                :
                m_Position(a_Position),
                m_Up(a_Up),
                m_Right(a_Right),
                m_Forward(a_Forward)
            {}

            CPU_ONLY ~DeviceCameraData() = default;

            unsigned int m_PixelIndex;
            float3 m_Position;
            float3 m_Up;
            float3 m_Right;
            float3 m_Forward;

        };

        CPU_ONLY PrimRayGenLaunchParameters(
            const uint2& a_Resolution,
            const DeviceCameraData& a_Camera,
            AtomicBuffer < WaveFront::IntersectionRayData>* a_PrimaryRays,
            const unsigned int a_FrameCount)
            :
            m_Resolution(a_Resolution),
            m_Camera(a_Camera),
            m_PrimaryRays(a_PrimaryRays),
    		m_FrameCount(a_FrameCount)
        {}

        CPU_ONLY ~PrimRayGenLaunchParameters() = default;

        const uint2 m_Resolution;
        const DeviceCameraData m_Camera;
        AtomicBuffer<IntersectionRayData> * m_PrimaryRays;
        const unsigned int m_FrameCount;
    };

    struct ShadingLaunchParameters
    {

        CPU_ONLY ShadingLaunchParameters(
            const uint3& a_ResolutionAndDepth,
            const SurfaceData* const a_CurrentSurfaceData,
            const SurfaceData* const a_TemporalSurfaceData,
            const VolumetricData* const a_CurrentVolumetricData,
            const cudaSurfaceObject_t const a_MotionVectorBuffer,
            const MemoryBuffer* const a_LightDataBuffer,
            const OptixTraversableHandle a_OptixSceneHandle,
            const OptixWrapper* const a_OptixWrapper,
            const unsigned a_CurrentDepth,
            const unsigned a_Seed,
            AtomicBuffer<IntersectionRayData>* a_RayBuffer,
            AtomicBuffer<ShadowRayData>* a_SolidShadowRayBuffer,
            AtomicBuffer<ShadowRayData>* a_VolumetricShadowRayBuffer,
            ReSTIR* a_ReSTIR,
            std::array<cudaSurfaceObject_t, static_cast<unsigned>(LightChannel::NUM_CHANNELS)> a_OutputChannels
        )
            :
        m_ResolutionAndDepth(a_ResolutionAndDepth),
        m_CurrentSurfaceData(a_CurrentSurfaceData),
        m_TemporalSurfaceData(a_TemporalSurfaceData),
        m_CurrentVolumetricData(a_CurrentVolumetricData),
        m_MotionVectorBuffer(a_MotionVectorBuffer),
        m_LightDataBuffer(a_LightDataBuffer),
        m_OptixSceneHandle(a_OptixSceneHandle),
        m_OptixWrapper(a_OptixWrapper),
        m_CurrentDepth(a_CurrentDepth),
        m_Seed(a_Seed),
        m_RayBuffer(a_RayBuffer),
        m_SolidShadowRayBuffer(a_SolidShadowRayBuffer),
        m_VolumetricShadowRayBuffer(a_VolumetricShadowRayBuffer),
        m_ReSTIR(a_ReSTIR),
        m_OutputChannels(a_OutputChannels)
        {}


        CPU_ONLY ~ShadingLaunchParameters() = default;

        const uint3 m_ResolutionAndDepth;
        const SurfaceData* const m_CurrentSurfaceData;
        const SurfaceData* const m_TemporalSurfaceData;
        const VolumetricData* const m_CurrentVolumetricData;
        const cudaSurfaceObject_t const m_MotionVectorBuffer;
        const MemoryBuffer* const m_LightDataBuffer;
        const OptixTraversableHandle m_OptixSceneHandle;
        const OptixWrapper* const m_OptixWrapper;
        const unsigned m_CurrentDepth;
        const unsigned m_Seed;
        AtomicBuffer<IntersectionRayData>* m_RayBuffer;
        AtomicBuffer<ShadowRayData>* m_SolidShadowRayBuffer;
        AtomicBuffer<ShadowRayData>* m_VolumetricShadowRayBuffer;
        ReSTIR* m_ReSTIR;
        std::array<cudaSurfaceObject_t, static_cast<unsigned>(LightChannel::NUM_CHANNELS)> m_OutputChannels;
        FrameStats* m_FrameStats;
    };

    struct WriteOutputParams {
        CPU_ONLY WriteOutputParams(
            const uint2& a_OutputResolution,
            const cudaSurfaceObject_t a_PixelBufferSingleChannel,
            uchar4* const a_FinalOutput)
            :
            m_OutputResolution(a_OutputResolution),
            m_PixelBufferSingleChannel(a_PixelBufferSingleChannel),
            m_FinalOutput(a_FinalOutput)
        {}

        CPU_ONLY ~WriteOutputParams() = default;

        const uint2& m_OutputResolution;
        const cudaSurfaceObject_t m_PixelBufferSingleChannel;
        uchar4* const m_FinalOutput;
    };


    struct PostProcessLaunchParameters
    {

        CPU_ONLY PostProcessLaunchParameters(
            const uint2& a_RenderResolution,
            const uint2& a_OutputResolution,
            const std::array<cudaSurfaceObject_t, static_cast<unsigned>(LightChannel::NUM_CHANNELS)> a_PixelBufferMultiChannel,
            const cudaSurfaceObject_t a_PixelBufferSingleChannel,
            uchar4* const a_FinalOutput,
            const bool a_BlendOutput,
            const unsigned a_BlendCount
        )
            :
            m_RenderResolution(a_RenderResolution),
            m_OutputResolution(a_OutputResolution),
            m_PixelBufferMultiChannel(a_PixelBufferMultiChannel),
            m_PixelBufferSingleChannel(a_PixelBufferSingleChannel),
            m_FinalOutput(a_FinalOutput),
            m_BlendOutput(a_BlendOutput),
            m_BlendCount(a_BlendCount)
        {}

        CPU_ONLY ~PostProcessLaunchParameters() = default;

        const uint2& m_RenderResolution;
        const uint2& m_OutputResolution;
        const std::array<cudaSurfaceObject_t, static_cast<unsigned>(LightChannel::NUM_CHANNELS)> m_PixelBufferMultiChannel;
        const cudaSurfaceObject_t m_PixelBufferSingleChannel;
        uchar4* const m_FinalOutput;
        bool m_BlendOutput;
        unsigned m_BlendCount;
    };


    struct OptixDenoiserLaunchParameters
    {

        CPU_ONLY OptixDenoiserLaunchParameters(
            const uint2& a_RenderResolution,
            const SurfaceData* a_CurrentSurfaceData,
            const cudaSurfaceObject_t a_PixelBufferSingleChannel,
            float3* a_IntermediaryInput,
            float3* a_AlbedoInput,
            float3* a_NormalInput,
            float2* a_FlowInput,
            float3* a_IntermediaryOutput
        )
            :
            m_RenderResolution(a_RenderResolution),
            m_CurrentSurfaceData(a_CurrentSurfaceData),
            m_PixelBufferSingleChannel(a_PixelBufferSingleChannel),
            m_IntermediaryInput(a_IntermediaryInput),
            m_AlbedoInput(a_AlbedoInput),
            m_NormalInput(a_NormalInput),
            m_FlowInput(a_FlowInput),
            m_IntermediaryOutput(a_IntermediaryOutput)
        {}

        CPU_ONLY ~OptixDenoiserLaunchParameters() = default;

        const uint2& m_RenderResolution;
        const SurfaceData* m_CurrentSurfaceData;
        const cudaSurfaceObject_t m_PixelBufferSingleChannel;
        float3* m_IntermediaryInput;
        float3* m_AlbedoInput;
        float3* m_NormalInput;
        float2* m_FlowInput;
        float3* m_IntermediaryOutput;
        float3* m_BlendOutput;
        bool m_UseBlendOutput;
        unsigned int m_BlendCount;
    };
}
