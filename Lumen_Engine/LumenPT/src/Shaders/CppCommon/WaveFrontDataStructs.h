#pragma once

#include "Cuda/cuda.h"

/*TODO: check if usage of textures can benefit performance
 *for buffers like intersectionBuffer or OutputBuffer.
*/

namespace WaveFront
{

    struct PixelBuffer
    {

        enum class OutputChannel : unsigned int
        {
            DIRECT,
            INDIRECT,
            SPECULAR,
            NUM_CHANNELS
        };

        __global__ PixelBuffer(unsigned int a_Size);

        //Ready only
        constexpr static unsigned int m_NumOutputChannels = static_cast<unsigned>(OutputChannel::NUM_CHANNELS);
        const unsigned int m_Size;

        //Read/Write
        float3 m_Pixels[][m_NumOutputChannels];

    };

    struct RayData
    {

        RayData(
            const float3& a_Origin, 
            const float3& a_Direction, 
            const float3& a_Contribution);

        //Read only
        float3 m_Origin;
        float3 m_Direction;
        float3 m_Contribution;

    };

    struct RayBatch
    {

        RayBatch(unsigned int a_Size);

        //Read only
        const unsigned int m_Size;

        //Read/Write
        RayData m_Rays[];

        void AddRay(const RayData& a_data, unsigned int a_index)
        {
            if (a_index < m_Size)
            {
                m_Rays[a_index] = a_data;
            }
        }

    };

    struct ShadowRayData
    {

        ShadowRayData(
            const float3& a_Origin,
            const float3& a_Direction,
            const float& a_MaxDistance,
            const float3& a_PotentialRadiance,
            PixelBuffer::OutputChannel a_OutputChannelIndex);

        //Read only
        float3 m_Origin;
        float3 m_Direction;
        float m_MaxDistance;
        float3 m_PotentialRadiance;
        PixelBuffer::OutputChannel m_OutputChannelIndex;

    };

    struct ShadowRayBatch
    {

        ShadowRayBatch(unsigned int a_Size);

        //Read only
        const unsigned int m_Size;

        //Read/Write
        ShadowRayData m_ShadowRays[];

    };

    struct IntersectionData
    {

        IntersectionData(
            unsigned int a_RayId,
            float a_IntersectionT,
            unsigned int m_TriangleId,
            unsigned int m_MeshId);

        //Read only
        unsigned int m_RayId;
        float m_IntersectionT;
        unsigned int m_TriangleId;
        unsigned int m_MeshId;

    };

    struct IntersectionBuffer
    {

        IntersectionBuffer(unsigned int a_Size);

        //Read only
        const unsigned int m_Size;

        //Read/Write
        IntersectionData m_Intersections[];

    };

    struct ResultBuffer
    {

        ResultBuffer(
            const RayBatch* a_PrimaryRays,
            const IntersectionBuffer* a_PrimaryIntersections,
            PixelBuffer* a_PixelOutput);

        //Read only
        const RayBatch* m_PrimaryRays;
        const IntersectionBuffer* m_PrimaryIntersections;

        //Read/Write
        PixelBuffer* m_PixelOutput;

    };

    struct DeviceCameraData
    {

        DeviceCameraData(
            const float3& a_Position, 
            const float3& a_Up, 
            const float3& a_Right, 
            const float3& a_Forward);
        ~DeviceCameraData();

        float3 m_Position;
        float3 m_Up;
        float3 m_Right;
        float3 m_Forward;

    };

    struct SetupLaunchParameters
    {

        SetupLaunchParameters(const uint2& a_Resolution, const DeviceCameraData& a_Camera);
        ~SetupLaunchParameters();

        const uint2 m_Resolution;
        const DeviceCameraData m_Camera;

    };

    struct ShadingLaunchParameters
    {

        ShadingLaunchParameters(
            const uint2& a_Resolution,
            const ResultBuffer* a_PrevOutput,
            const IntersectionBuffer* a_Intersections,
            RayBatch* a_SecondaryRays,
            ShadowRayBatch* a_ShadowRayBatches[]);

        ~ShadingLaunchParameters();

        //Read only
        const uint2 m_Resolution;
        const ResultBuffer* m_PrevOutput;
        const IntersectionBuffer* m_Intersections;
        //TODO: Geometry buffer
        //TODO: Light buffer

        //Write
        RayBatch* m_SecondaryRays;
        ShadowRayBatch* m_ShadowRaysBatches[];

    };

    struct PostProcessLaunchParameters
    {

        PostProcessLaunchParameters(
            const uint2& a_Resolution, 
            const ResultBuffer* a_WavefrontOutput, 
            char4* a_ImageOutput);
        ~PostProcessLaunchParameters();

        //Read only
        const uint2 m_Resolution;
        const ResultBuffer* m_WavefrontOutputBuffer;

        //Read/Write
        char4* m_ImageOutputBuffer;

    };

}