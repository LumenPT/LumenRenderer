#pragma once

/*TODO: check if usage of textures can benefit performance
 *for buffers like intersectionBuffer or OutputBuffer.
*/

namespace WaveFront
{

    struct RayData
    {

        const float3 m_Origin;
        const float3 m_Direction;
        const float3 m_Contribution;

    };

    struct RayBatch
    {

        const unsigned int m_Size;
        RayData m_Rays[];

    };

    struct ShadowRayData
    {

        const float3 m_Origin;
        const float3 m_Direction;
        const float m_MaxDistance;
        const float3 m_PotentialRadiance;
        unsigned int m_OutputChannelIndex;

    };

    struct ShadowRayBatch
    {

        const unsigned int m_Size;
        ShadowRayData m_ShadowRays[];

    };

    struct IntersectionData
    {

        unsigned int m_RayId;
        float m_IntersectionT;
        unsigned int m_TriangleId;
        unsigned int m_MeshId;

    };

    struct IntersectionBuffer
    {

        const unsigned int m_Size;
        IntersectionData m_Intersections[];

    };

    struct OutputBuffer
    {

        const static unsigned int m_NumOutputChannels = 3;
        const unsigned int m_Size;
        float3 m_Pixels[][m_NumOutputChannels];

    };

    struct PixelBuffer
    {

        RayBatch* m_PrimaryRays;
        IntersectionBuffer* m_PrimaryIntersections;
        OutputBuffer* m_PixelOutput;

    };

    struct ShadingLaunchParameters
    {

        PixelBuffer* m_Output;
        PixelBuffer* m_PrevOutput;
        RayBatch* m_SecondaryRays;
        IntersectionBuffer* m_Intersections;
        const float2 m_OutputSize;
        ShadowRayBatch* m_ShadowRaysBatches[];

    };

    struct DeviceCameraData
    {

        float3 m_Position;
        float3 m_Up;
        float3 m_Right;
        float3 m_Forward;

    };

    struct PrimRayGenLaunchParameters
    {

        DeviceCameraData m_Camera;
        float2 m_OutputSize;

    };

}