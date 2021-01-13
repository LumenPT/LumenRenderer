#pragma once

#include "Cuda/cuda.h"

/*TODO: check if usage of textures can benefit performance
 *for buffers like intersectionBuffer or OutputBuffer.
*/

#define CPU_ONLY __host__
#define CPU_GPU __host__ __device__
#define GPU_ONLY __device__

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

        CPU_ONLY PixelBuffer(unsigned int a_Size)
            :
        m_Size(a_Size)
        {
            cudaMalloc(reinterpret_cast<void**>(&m_Pixels), m_Size * sizeof(float3) * m_NumOutputChannels);
        }

        CPU_ONLY ~PixelBuffer()
        {
            cudaFree(m_Pixels);
        }

        //Ready only
        constexpr static unsigned int m_NumOutputChannels = static_cast<unsigned>(OutputChannel::NUM_CHANNELS);
        const unsigned int m_Size;

        //Read/Write
        float3 m_Pixels[][m_NumOutputChannels];

    };

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

        CPU_ONLY RayBatch(unsigned int a_Size)
            :
        m_Size(a_Size)
        {
            cudaMalloc(reinterpret_cast<void**>(&m_Rays), m_Size * sizeof(RayData));
        }

        CPU_ONLY ~RayBatch()
        {
            cudaFree(m_Rays);
        }

        //Read only
        const unsigned int m_Size;

        //Read/Write
        RayData m_Rays[];

        GPU_ONLY void AddRay(const RayData& a_data, unsigned int a_index)
        {
            if (a_index < m_Size)
            {
                m_Rays[a_index] = a_data;
            }
        }

    };

    struct ShadowRayData
    {

        CPU_GPU ShadowRayData()
            :
        m_Origin(make_float3(0.f, 0.f, 0.f)),
        m_Direction(make_float3(0.f, 0.f, 0.f)),
        m_MaxDistance(0.f),
        m_PotentialRadiance(make_float3(0.f, 0.f, 0.f)),
        m_OutputChannelIndex(PixelBuffer::OutputChannel::DIRECT)
        {}

        CPU_GPU ShadowRayData(
            const float3& a_Origin,
            const float3& a_Direction,
            const float& a_MaxDistance,
            const float3& a_PotentialRadiance,
            PixelBuffer::OutputChannel a_OutputChannelIndex)
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
        PixelBuffer::OutputChannel m_OutputChannelIndex;

    };

    struct ShadowRayBatch
    {

        CPU_ONLY ShadowRayBatch(unsigned int a_Size)
            :
        m_Size(a_Size)
        {
            cudaMalloc(reinterpret_cast<void**>(&m_ShadowRays), m_Size * sizeof(ShadowRayData));
        }

        CPU_ONLY ~ShadowRayBatch()
        {
            cudaFree(m_ShadowRays);
        }

        //Read only
        const unsigned int m_Size;

        //Read/Write
        ShadowRayData m_ShadowRays[];

    };

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

        CPU_ONLY IntersectionBuffer(unsigned int a_Size)
            :
        m_Size(a_Size)
        {
            cudaMalloc(reinterpret_cast<void**>(&m_Intersections), m_Size * sizeof(IntersectionData));
        }

        CPU_ONLY ~IntersectionBuffer()
        {
            cudaFree(m_Intersections);
        }

        //Read only
        const unsigned int m_Size;

        //Read/Write
        IntersectionData m_Intersections[];

    };

    struct ResultBuffer
    {

        CPU_ONLY ResultBuffer(
            const RayBatch* a_PrimaryRays,
            const IntersectionBuffer* a_PrimaryIntersections,
            PixelBuffer* a_PixelOutput)
            :
        m_PrimaryRays(a_PrimaryRays),
        m_PrimaryIntersections(a_PrimaryIntersections),
        m_PixelOutput(a_PixelOutput)
        {}

        CPU_ONLY ~ResultBuffer()
        {
            m_PrimaryRays = nullptr;
            m_PrimaryIntersections = nullptr;
            m_PixelOutput = nullptr;
        }

        //Read only
        const RayBatch* m_PrimaryRays;
        const IntersectionBuffer* m_PrimaryIntersections;

        //Read/Write
        PixelBuffer* m_PixelOutput;

    };

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

    struct SetupLaunchParameters
    {

        CPU_ONLY SetupLaunchParameters(
            const uint2& a_Resolution, 
            const DeviceCameraData& a_Camera)
            :
        m_Resolution(a_Resolution),
        m_Camera(a_Camera)
        {}

        CPU_ONLY ~SetupLaunchParameters() = default;

        const uint2 m_Resolution;
        const DeviceCameraData m_Camera;

    };

    struct ShadingLaunchParameters
    {

        CPU_ONLY ShadingLaunchParameters(
            const uint2& a_Resolution,
            const ResultBuffer* a_PrevOutput,
            const IntersectionBuffer* a_Intersections,
            RayBatch* a_SecondaryRays,
            ShadowRayBatch* a_ShadowRayBatches[],
            LightBuffer* a_Lights);

        ~ShadingLaunchParameters();

        //Read only
        const uint2 m_Resolution;
        const ResultBuffer* m_PrevOutput;
        const IntersectionBuffer* m_Intersections;
        //TODO: Geometry buffer
        //TODO: Light buffer
        const LightBuffer* m_LightBuffer;

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