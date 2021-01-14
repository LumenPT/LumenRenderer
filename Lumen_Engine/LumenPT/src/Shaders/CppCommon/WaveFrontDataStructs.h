#pragma once

#include <Optix/optix.h>
#include <Cuda/cuda/helpers.h>

/*TODO: check if usage of textures can benefit performance
 *for buffers like intersectionBuffer or OutputBuffer.
*/

#define CPU_ONLY __host__
#define CPU_GPU __host__ __device__
#define GPU_ONLY __device__

namespace WaveFront
{

    //Scene data

    struct LightData    //placeholder contents
    {
        LightData(float a_Intensity, float3 a_Color);

        // material or something?
        float m_Intensity;
        float3 m_Color;
        float3 m_Position;
    };

    struct LightBuffer  //placeholder contents
    {
        LightBuffer(unsigned int a_Size);

        const unsigned int m_Size;
        LightData m_Lights[];
    };

    struct MeshData
    {

        //Buffer handles
        //Buffer vertex layouts

    };

    struct MeshBuffer
    {

        const unsigned int m_Size;
        MeshData m_Meshes[];

    };



    //Ray & Intersection data and buffers

    struct RayData
    {

        CPU_GPU RayData()
            :
        m_Origin(make_float3(0.f,0.f,0.f)),
        m_Direction(make_float3(0.f, 0.f, 0.f)),
        m_Contribution(make_float3(0.f, 0.f, 0.f))
        {}

        CPU_GPU RayData(
            const float3& a_Origin, 
            const float3& a_Direction, 
            const float3& a_Contribution)
            :
        m_Origin(a_Origin),
        m_Direction(a_Direction),
        m_Contribution(a_Contribution)
        {}

        //Read only
        float3 m_Origin;
        float3 m_Direction;
        float3 m_Contribution;

    };

    struct RayBatch
    {

        RayBatch()
            :
        m_NumPixels(0u),
        m_RaysPerPixel(0u),
        m_Rays()
        {}

        //Read only
        const unsigned int m_NumPixels;
        const unsigned int m_RaysPerPixel;

        //Read/Write
        RayData m_Rays[];

        CPU_GPU unsigned int GetSize() const
        {
            return m_NumPixels * m_RaysPerPixel;
        }

        GPU_ONLY void SetRay(
            const RayData& a_Data, 
            unsigned int a_PixelIndex,
            unsigned int a_RayIndex)
        {
            if( a_PixelIndex < m_NumPixels && 
                a_RayIndex < m_RaysPerPixel)
            {
                m_Rays[a_PixelIndex * m_RaysPerPixel + a_RayIndex] = a_Data;
            }
        }

    };

    struct IntersectionData
    {

        CPU_GPU IntersectionData()
            :
        m_RayId(0),
        m_IntersectionT(-1.f),
        m_TriangleId(0),
        m_MeshId(0)
        {}
        

        CPU_GPU IntersectionData(
            unsigned int a_RayId,
            float a_IntersectionT,
            unsigned int a_TriangleId,
            unsigned int a_MeshId)
            :
        m_RayId(a_RayId),
        m_IntersectionT(a_IntersectionT),
        m_TriangleId(a_TriangleId),
        m_MeshId(a_MeshId)
        {}

        CPU_GPU bool IsIntersection() const
        {
            return (m_IntersectionT > 0.f);
        }

        //Read only
        unsigned int m_RayId;
        float m_IntersectionT;
        unsigned int m_TriangleId;
        unsigned int m_MeshId;

    };

    struct IntersectionBuffer
    {

        IntersectionBuffer()
            :
        m_NumPixels(0u),
        m_IntersectionsPerPixel(0u),
        m_Intersections()
        {}

        //Read only
        const unsigned int m_NumPixels;
        const unsigned int m_IntersectionsPerPixel;

        //Read/Write
        IntersectionData m_Intersections[];

        CPU_GPU unsigned int GetSize() const
        {
            return m_NumPixels * m_IntersectionsPerPixel;
        }

        CPU_GPU void SetIntersection(
            const IntersectionData& a_Data, 
            unsigned int a_PixelIndex, 
            unsigned int a_IntersectionIndex)
        {

            if( a_PixelIndex < m_NumPixels && 
                a_IntersectionIndex < m_IntersectionsPerPixel)
            {
                m_Intersections[a_PixelIndex * m_IntersectionsPerPixel + a_IntersectionIndex] = a_Data;
            }

        }

    };

    struct PixelBuffer
    {

        PixelBuffer()
            :
        m_NumPixels(0u),
        m_Pixels()
        {}

        //Ready only
        const unsigned int m_NumPixels;

        //Read/Write
        float3 m_Pixels[];

        CPU_GPU void SetPixel(const float3& a_value, unsigned int a_PixelIndex)
        {
            if(a_PixelIndex < m_NumPixels)
            {
                m_Pixels[a_PixelIndex] = a_value;
            }
        }

    };

    struct ResultBuffer
    {

        enum class OutputChannel : unsigned int
        {
            DIRECT,
            INDIRECT,
            SPECULAR,
            NUM_CHANNELS
        };

        ResultBuffer()
            :
        //m_PrimaryRays(nullptr),
        //m_PrimaryIntersections(nullptr),
        m_PixelOutput()
        {}

        //Read only (Might not be necessary to store the primary rays and intersections here)
        //const RayBatch* const m_PrimaryRays;
        //const IntersectionBuffer* const m_PrimaryIntersections;
        constexpr static unsigned int s_NumOutputChannels = static_cast<unsigned>(OutputChannel::NUM_CHANNELS);

        //Read/Write
        PixelBuffer* const m_PixelOutput[s_NumOutputChannels];

    };

    struct ShadowRayData
    {

        CPU_GPU ShadowRayData()
            :
            m_Origin(make_float3(0.f, 0.f, 0.f)),
            m_Direction(make_float3(0.f, 0.f, 0.f)),
            m_MaxDistance(0.f),
            m_PotentialRadiance(make_float3(0.f, 0.f, 0.f)),
            m_OutputChannelIndex(ResultBuffer::OutputChannel::DIRECT)
        {}

        CPU_GPU ShadowRayData(
            const float3& a_Origin,
            const float3& a_Direction,
            const float& a_MaxDistance,
            const float3& a_PotentialRadiance,
            ResultBuffer::OutputChannel a_OutputChannelIndex)
            :
            m_Origin(a_Origin),
            m_Direction(a_Direction),
            m_MaxDistance(a_MaxDistance),
            m_PotentialRadiance(a_PotentialRadiance),
            m_OutputChannelIndex(a_OutputChannelIndex)
        {}

        //Read only
        float3 m_Origin;
        float3 m_Direction;
        float m_MaxDistance;
        float3 m_PotentialRadiance;
        ResultBuffer::OutputChannel m_OutputChannelIndex;

    };

    struct ShadowRayBatch
    {

        ShadowRayBatch()
            :
        m_MaxDepth(0u),
        m_NumPixels(0u),
        m_RaysPerPixel(0u),
        m_ShadowRays()
        {}

        //Read only
        const unsigned int m_MaxDepth;
        const unsigned int m_NumPixels;
        const unsigned int m_RaysPerPixel;

        //Read/Write
        ShadowRayData m_ShadowRays[];

        CPU_GPU unsigned int GetSize() const
        {
            return m_MaxDepth * m_NumPixels * m_RaysPerPixel;
        }

        CPU_GPU void SetShadowRay(
            const ShadowRayData& a_Data, 
            unsigned int a_DepthIndex, 
            unsigned int a_PixelIndex, 
            unsigned int a_RayIndex)
        {
            if( a_DepthIndex < m_MaxDepth &&
                a_PixelIndex < m_NumPixels && 
                a_RayIndex < m_RaysPerPixel)
            {
                m_ShadowRays[a_DepthIndex * m_NumPixels * m_RaysPerPixel + a_PixelIndex * m_RaysPerPixel + a_RayIndex] = a_Data;
            }
        }

    };



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

        float3 m_Position;
        float3 m_Up;
        float3 m_Right;
        float3 m_Forward;

    };



    //Kernel Launch parameters

    struct SetupLaunchParameters
    {

        CPU_ONLY SetupLaunchParameters(
            const uint2& a_Resolution, 
            const DeviceCameraData& a_Camera,
            RayBatch* const a_PrimaryRays)
            :
        m_Resolution(a_Resolution),
        m_Camera(a_Camera),
        m_PrimaryRays(a_PrimaryRays)
        {}

        CPU_ONLY ~SetupLaunchParameters() = default;

        const uint2 m_Resolution;
        const DeviceCameraData m_Camera;
        RayBatch* const m_PrimaryRays;

    };

    struct ShadingLaunchParameters
    {

        CPU_ONLY ShadingLaunchParameters(
            const uint3& a_ResolutionAndDepth,
            const RayBatch* const a_PrimaryRays,
            const IntersectionBuffer* a_PrimaryIntersections,
            const RayBatch* const a_PrevRays,
            const IntersectionBuffer* a_Intersections,
            RayBatch* a_SecondaryRays,
            ShadowRayBatch* a_ShadowRayBatch,
            const LightBuffer* a_Lights)
            :
        m_ResolutionAndDepth(a_ResolutionAndDepth),
        m_PrimaryRays(a_PrimaryRays),
        m_PrimaryIntersections(a_PrimaryIntersections),
        m_PrevRays(a_PrevRays),
        m_Intersections(a_Intersections),
        m_LightBuffer(a_Lights),
        m_SecondaryRays(a_SecondaryRays),
        m_ShadowRaysBatch(a_ShadowRayBatch)
        {}

        CPU_ONLY ~ShadingLaunchParameters() = default;

        //Read only
        const uint3 m_ResolutionAndDepth;
        const RayBatch* const m_PrimaryRays;
        const IntersectionBuffer* const m_PrimaryIntersections;
        const RayBatch* const m_PrevRays;
        const IntersectionBuffer* const m_Intersections;
        //TODO: Geometry buffer
        //TODO: Light buffer
        const LightBuffer* const m_LightBuffer;

        //Write
        RayBatch* const m_SecondaryRays;
        ShadowRayBatch* const m_ShadowRaysBatch;

    };

    struct PostProcessLaunchParameters
    {

        CPU_ONLY PostProcessLaunchParameters(
            const uint2& a_Resolution, 
            const ResultBuffer* const a_WavefrontOutput,
            PixelBuffer* const a_MergedResults,
            char4* const a_ImageOutput)
            :
        m_Resolution(a_Resolution),
        m_WavefrontOutput(a_WavefrontOutput),
        m_MergedResults(a_MergedResults),
        m_ImageOutput(a_ImageOutput)
        {}

        CPU_ONLY ~PostProcessLaunchParameters() = default;

        //Read only
        const uint2 m_Resolution;
        const ResultBuffer* const m_WavefrontOutput;

        //Read/Write
        PixelBuffer* const m_MergedResults; //Used to merge results from multiple channels into one channel.
        char4* const m_ImageOutput; //Used to display image after DLSS algorithm has run on merged results.

    };



    // Shader Launch parameters

    struct CommonOptixLaunchParameters
    {

        OptixTraversableHandle m_Solids;
        OptixTraversableHandle m_Volumes;

    };

    struct ResolveRaysLaunchParameters
    {

        uint3 m_ResolutionAndDepth;
        RayBatch* m_Rays;
        IntersectionBuffer* m_Intersections;

    };

    struct ResolveShadowRaysLaunchParameters
    {


        uint3 m_ResolutionAndDepth;
        ShadowRayBatch* m_ShadowRays;
        ResultBuffer* m_Results;

    };

}