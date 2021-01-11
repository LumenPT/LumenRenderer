#pragma once

/*TODO: check if usage of textures can benefit performance
 *for buffers like intersectionBuffer or OutputBuffer.
*/

namespace WaveFront
{

    struct RayData
    {

        //Read only
        const float3 m_Origin;
        const float3 m_Direction;
        const float3 m_Contribution;

    };

    struct RayBatch
    {

        //Read only
        const unsigned int m_Size;

        //Read/Write
        RayData m_Rays[];

    };

    struct ShadowRayData
    {

        //Read only
        const float3 m_Origin;
        const float3 m_Direction;
        const float m_MaxDistance;
        const float3 m_PotentialRadiance;
        const unsigned int m_OutputChannelIndex;

    };

    struct ShadowRayBatch
    {

        //Read only
        const unsigned int m_Size;

        //Read/Write
        ShadowRayData m_ShadowRays[];

    };

    struct IntersectionData
    {

        //Read only
        const unsigned int m_RayId;
        const float m_IntersectionT;
        const unsigned int m_TriangleId;
        const unsigned int m_MeshId;

    };

    struct IntersectionBuffer
    {

        //Read only
        const unsigned int m_Size;

        //Read/Write
        IntersectionData m_Intersections[];

    };

    struct PixelBuffer
    {

        enum class OutputChannel : unsigned int
        {
            DIRECT,
            INDIRECT,
            SPECULAR,
            NUM_CHANNELS
        };

        //Ready only
        constexpr static unsigned int m_NumOutputChannels = static_cast<unsigned>(OutputChannel::NUM_CHANNELS);
        const unsigned int m_Size;

        //Read/Write
        float3 m_Pixels[][m_NumOutputChannels];

    };

    struct OutputBuffer
    {

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

        SetupLaunchParameters();
        ~SetupLaunchParameters();

        const uint2 m_Resolution;
        const DeviceCameraData m_Camera;

    };

    struct ShadingLaunchParameters
    {

        ShadingLaunchParameters(
            const uint2& a_Resolution,
            const OutputBuffer* a_PrevOutput,
            const IntersectionBuffer* a_Intersections,
            RayBatch* a_SecondaryRays,
            ShadowRayBatch* a_ShadowRayBatches[]);

        ~ShadingLaunchParameters();

        //Read only
        const uint2 m_Resolution;
        const OutputBuffer* m_PrevOutput;
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
            const OutputBuffer* a_WavefrontOutput, 
            char4* a_ImageOutput);
        ~PostProcessLaunchParameters();

        //Read only
        const uint2 m_Resolution;
        const OutputBuffer* m_WavefrontOutputBuffer;

        //Read/Write
        char4* m_ImageOutputBuffer;

    };

}