#pragma once
#include "../WaveFrontDataStructs.h"
#include "../CudaDefines.h"
#include "SurfaceDataBuffer.h"

namespace WaveFront
{
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
            const SurfaceData* a_CurrentSurfaceData,
            const SurfaceData* a_TemporalSurfaceData,
            AtomicBuffer<ShadowRayData>* a_ShadowRays,
            TriangleLight* a_TriangleLights,
            std::uint32_t a_NumLights,
            const CDF* const a_CDF = nullptr,
            float3* a_Output = nullptr
        ) :
        m_ResolutionAndDepth(a_ResolutionAndDepth),
        m_CurrentSurfaceData(a_CurrentSurfaceData),
        m_TemporalSurfaceData(a_TemporalSurfaceData),
        m_ShadowRays(a_ShadowRays),
        m_TriangleLights(a_TriangleLights),
        m_NumLights(a_NumLights),
        m_CDF(a_CDF),
        m_Output(a_Output)
        {}


        CPU_ONLY ~ShadingLaunchParameters() = default;

        //Read only
        const uint3 m_ResolutionAndDepth;
        const SurfaceData* const m_CurrentSurfaceData;
        const SurfaceData* const m_TemporalSurfaceData;
        const TriangleLight* const m_TriangleLights;
        const std::uint32_t m_NumLights;
        const CDF* const m_CDF;

        float3* m_Output;
        AtomicBuffer<ShadowRayData>* m_ShadowRays;
    };

    struct PostProcessLaunchParameters
    {

        CPU_ONLY PostProcessLaunchParameters(
            const uint2& a_RenderResolution,
            const uint2& a_OutputResolution,
            const float3* const a_WavefrontOutput,
            float3* const a_ProcessedOutput,
            uchar4* const a_FinalOutput)
            :
            m_RenderResolution(a_RenderResolution),
            m_OutputResolution(a_OutputResolution),
            m_WavefrontOutput(a_WavefrontOutput),
            m_ProcessedOutput(a_ProcessedOutput),
            m_FinalOutput(a_FinalOutput)
        {}

        CPU_ONLY ~PostProcessLaunchParameters() = default;

        const uint2& m_RenderResolution;
        const uint2& m_OutputResolution;
        const float3* const m_WavefrontOutput;
        float3* const m_ProcessedOutput;
        uchar4* const m_FinalOutput;
    };

}
