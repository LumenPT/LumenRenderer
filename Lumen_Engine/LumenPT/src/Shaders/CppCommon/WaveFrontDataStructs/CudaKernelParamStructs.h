#pragma once
#include "../CudaDefines.h"

#include <Optix/optix.h>
#include <Cuda/cuda/helpers.h>
#include <cassert>

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
            IntersectionRayBatch* const a_PrimaryRays)
            :
            m_Resolution(a_Resolution),
            m_Camera(a_Camera),
            m_PrimaryRays(a_PrimaryRays)
        {}

        CPU_ONLY ~PrimRayGenLaunchParameters() = default;

        const uint2 m_Resolution;
        const DeviceCameraData m_Camera;
        IntersectionRayBatch* const m_PrimaryRays;

    };

    struct ShadingLaunchParameters
    {

        CPU_ONLY ShadingLaunchParameters(
            const uint3& a_ResolutionAndDepth,
            const IntersectionRayBatch* const a_PrimaryRaysPrevFrame,
            const IntersectionBuffer* a_PrimaryIntersectionsPrevFrame,
            const IntersectionRayBatch* const a_CurrentRays,
            const IntersectionBuffer* a_CurrentIntersections,
            IntersectionRayBatch* a_SecondaryRays,
            ShadowRayBatch* a_ShadowRayBatch,
            const LightBuffer* a_Lights,
            CDF* const a_CDF = nullptr,
            ResultBuffer* a_DEBUGResultBuffer = nullptr)
            :
            m_ResolutionAndDepth(a_ResolutionAndDepth),
            m_PrimaryRaysPrevFrame(a_PrimaryRaysPrevFrame),
            m_PrimaryIntersectionsPrevFrame(a_PrimaryIntersectionsPrevFrame),
            m_CurrentRays(a_CurrentRays),
            m_CurrentIntersections(a_CurrentIntersections),
            m_LightBuffer(a_Lights),
            m_SecondaryRays(a_SecondaryRays),
            m_ShadowRaysBatch(a_ShadowRayBatch),
            m_CDF(a_CDF),
            m_DEBUGResultBuffer(a_DEBUGResultBuffer)
        {}

        CPU_ONLY ~ShadingLaunchParameters() = default;

        //Read only
        const uint3 m_ResolutionAndDepth;
        const IntersectionRayBatch* const m_PrimaryRaysPrevFrame;
        const IntersectionBuffer* const m_PrimaryIntersectionsPrevFrame;
        const IntersectionRayBatch* const m_CurrentRays;
        const IntersectionBuffer* const m_CurrentIntersections;
        //TODO: Geometry buffer
        //TODO: Light buffer
        const LightBuffer* const m_LightBuffer;
        CDF* const m_CDF;

        //Write
        IntersectionRayBatch* const m_SecondaryRays;
        ShadowRayBatch* const m_ShadowRaysBatch;
        //TEMP DEBUG STUFF
        ResultBuffer* m_DEBUGResultBuffer;

    };

    struct PostProcessLaunchParameters
    {

        CPU_ONLY PostProcessLaunchParameters(
            const uint2& a_RenderResolution,
            const uint2& a_OutputResolution,
            const ResultBuffer* const a_WavefrontOutput,
            PixelBuffer* const a_MergedResults,
            uchar4* const a_ImageOutput)
            :
            m_RenderResolution(a_RenderResolution),
            m_OutputResolution(a_OutputResolution),
            m_WavefrontOutput(a_WavefrontOutput),
            m_MergedResults(a_MergedResults),
            m_ImageOutput(a_ImageOutput)
        {}

        CPU_ONLY ~PostProcessLaunchParameters() = default;

        //Read only
        const uint2 m_RenderResolution;
        const uint2 m_OutputResolution;
        const ResultBuffer* const m_WavefrontOutput;

        //Read/Write
        PixelBuffer* const m_MergedResults; //Used to merge results from multiple channels into one channel.
        uchar4* const m_ImageOutput; //Used to display image after DLSS algorithm has run on merged results.

    };

}
