#pragma once
#include "../WaveFrontDataStructs.h"
#include "../CudaDefines.h"
#include "MotionVectorsGenerationData.h"
#include "SurfaceDataBuffer.h"

class ReSTIR;

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
            const SurfaceData* a_CurrentSurfaceData,
            const SurfaceData* a_TemporalSurfaceData,
            const AtomicBuffer<IntersectionData>* a_Intersections,
            AtomicBuffer<IntersectionRayData>* a_RayBuffer,
            AtomicBuffer<ShadowRayData>* a_ShadowRays,
            TriangleLight* a_TriangleLights,
            std::uint32_t a_NumLights,
            const float3& a_CameraPosition,
            const float3& a_CameraDirection,
            const OptixTraversableHandle a_OptixSceneHandle,
            const WaveFront::OptixWrapper* a_OptixSystem,
            const unsigned a_Seed,
            ReSTIR* a_ReSTIR,
            unsigned a_Currentdepth,
            WaveFront::MotionVectorBuffer* a_MotionVectorBuffer,
            const unsigned a_NumIntersections,
            float3* a_Output = nullptr
        )
        :
        m_ResolutionAndDepth(a_ResolutionAndDepth),
        m_CurrentSurfaceData(a_CurrentSurfaceData),
        m_TemporalSurfaceData(a_TemporalSurfaceData),
        m_IntersectionData(a_Intersections),
        m_TriangleLights(a_TriangleLights),
        m_NumLights(a_NumLights),
        m_CurrentDepth(a_Currentdepth),
        m_CameraPosition(a_CameraPosition),
        m_CameraDirection(a_CameraDirection),
        m_OptixSceneHandle(a_OptixSceneHandle),
        m_OptixSystem(a_OptixSystem),
        m_Seed(a_Seed),
        m_NumIntersections(a_NumIntersections),
        m_RayBuffer(a_RayBuffer),
        m_ReSTIR(a_ReSTIR),
        m_Output(a_Output),
        m_ShadowRays(a_ShadowRays),
        m_MotionVectorBuffer(a_MotionVectorBuffer)
        {}


        CPU_ONLY ~ShadingLaunchParameters() = default;

        //Read only
        const uint3 m_ResolutionAndDepth;
        const SurfaceData* const m_CurrentSurfaceData;
        const SurfaceData* const m_TemporalSurfaceData;
        const AtomicBuffer<IntersectionData>* const m_IntersectionData;
        const TriangleLight* const m_TriangleLights;
        const std::uint32_t m_NumLights;
        const unsigned m_CurrentDepth;
        const float3 m_CameraPosition;
        const float3 m_CameraDirection;
        const OptixTraversableHandle m_OptixSceneHandle;
        const WaveFront::OptixWrapper* m_OptixSystem;
        const unsigned m_Seed;
        const unsigned m_NumIntersections;
        AtomicBuffer<IntersectionRayData>* m_RayBuffer;
        ReSTIR* m_ReSTIR;
        float3* m_Output;
        AtomicBuffer<ShadowRayData>* m_ShadowRays;
        WaveFront::MotionVectorBuffer* m_MotionVectorBuffer;
    };

    struct PostProcessLaunchParameters
    {

        CPU_ONLY PostProcessLaunchParameters(
            const uint2& a_RenderResolution,
            const uint2& a_OutputResolution,
            const float3* const a_WavefrontOutput,
            float3* const a_ProcessedOutput,
            uchar4* const a_FinalOutput,
            const bool a_BlendOutput,
            const unsigned a_BlendCount
        )
            :
            m_RenderResolution(a_RenderResolution),
            m_OutputResolution(a_OutputResolution),
            m_WavefrontOutput(a_WavefrontOutput),
            m_ProcessedOutput(a_ProcessedOutput),
            m_FinalOutput(a_FinalOutput),
            m_BlendOutput(a_BlendOutput),
            m_BlendCount(a_BlendCount)
        {}

        CPU_ONLY ~PostProcessLaunchParameters() = default;

        const uint2& m_RenderResolution;
        const uint2& m_OutputResolution;
        const float3* const m_WavefrontOutput;
        float3* const m_ProcessedOutput;
        uchar4* const m_FinalOutput;
        const bool m_BlendOutput;
        const unsigned m_BlendCount;
    };

}
