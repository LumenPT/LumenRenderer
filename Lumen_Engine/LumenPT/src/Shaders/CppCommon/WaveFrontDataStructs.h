#pragma once

#include <Optix/optix.h>
#include <Cuda/cuda/helpers.h>
#include "ModelStructs.h"
#include "ReSTIRData.h"
#include <assert.h>
#include <cstdio>
#include <array>

/*TODO: check if usage of textures can benefit performance
 *for buffers like intersectionBuffer or OutputBuffer.
*/

#if defined CPU_ONLY
#undef CPU_ONLY
#endif
#if defined CPU_GPU
#undef CPU_GPU
#endif
#if defined GPU_ONLY
#undef GPU_ONLY
#endif

#define CPU_ONLY __host__
#define CPU_GPU __host__ __device__
#define GPU_ONLY __device__
#define INLINE __forceinline__

namespace WaveFront
{

    //Scene data

    struct LightBuffer
    {

        CPU_GPU LightBuffer(unsigned int a_Size)
            :
        m_Size(a_Size),
        m_Lights()
        {}

        //TEMPORARY CONSTRUCTOR, not very good practice probs... TODO: Figure out if there is a way to make it better.
        CPU_ONLY LightBuffer(unsigned int a_Size, TriangleLight a_Lights[])
            :
        m_Size(a_Size),
        m_Lights()
        {

            m_LightPtr = reinterpret_cast<TriangleLight*>(malloc(static_cast<size_t>(m_Size) * sizeof(TriangleLight)));
            if(m_LightPtr != nullptr)
            {
                memcpy(m_LightPtr, a_Lights, static_cast<size_t>(m_Size) * sizeof(TriangleLight));
            }

        }

        const unsigned int m_Size;

        union
        {
            TriangleLight m_Lights[];
            TriangleLight* m_LightPtr;
        };

    };

    //Ray & Intersection data and buffers

    struct RayData
    {

        CPU_GPU RayData()
            :
        m_Origin(make_float3(0.f)),
        m_Direction(make_float3(0.f)),
        m_Contribution(make_float3(0.f))
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

        GPU_ONLY INLINE bool IsValidRay() const
        {

            return !(m_Direction.x == 0.f &&
                    m_Direction.y == 0.f &&
                    m_Direction.z == 0.f);

        }

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

        CPU_GPU INLINE unsigned int GetSize() const
        {
            return m_NumPixels * m_RaysPerPixel;
        }

        GPU_ONLY INLINE void SetRay(
            const RayData& a_Data, 
            unsigned int a_PixelIndex,
            unsigned int a_RayIndex)
        {
            /*RayData* rayPtr =  &m_Rays[GetRayIndex(a_PixelIndex, a_RayIndex)];
            printf("Buffer Start: %p, Buffer ptr: %p , Buffer Offset: %lli ", m_Rays, rayPtr, (reinterpret_cast<char*>(rayPtr) - reinterpret_cast<char*>(m_Rays)));*/

            m_Rays[GetRayIndex(a_PixelIndex, a_RayIndex)] = a_Data;
        }

        GPU_ONLY INLINE const RayData& GetRay(unsigned int a_PixelIndex, const unsigned int a_RayIndex) const
        {

            return m_Rays[GetRayIndex(a_PixelIndex, a_RayIndex)];

        }

        GPU_ONLY INLINE const RayData& GetRay(unsigned int a_RayIndex) const
        {
            assert(a_RayIndex < GetSize());

            return m_Rays[a_RayIndex];

        }

        //Gets an index to a Ray in the m_Rays array, taking into account the number of pixels and the number of rays per pixel.
        GPU_ONLY INLINE unsigned int GetRayIndex(unsigned int a_PixelIndex, const unsigned int a_RayIndex) const
        {

            assert(a_PixelIndex < m_NumPixels&& a_RayIndex < m_RaysPerPixel);

            return a_PixelIndex * m_RaysPerPixel + a_RayIndex;

        }

    };

    struct IntersectionData
    {

        CPU_GPU IntersectionData()
            :
        m_RayIndex(0),
        m_IntersectionT(-1.f),
        m_TriangleId(0),
        m_MeshAndInstanceId(0)
        {}
        

        CPU_GPU IntersectionData(
            unsigned int a_RayIndex,
            float a_IntersectionT,
            unsigned int a_TriangleId,
            unsigned int a_MeshAndInstanceId)
            :
        m_RayIndex(a_RayIndex),
        m_IntersectionT(a_IntersectionT),
        m_TriangleId(a_TriangleId),
        m_MeshAndInstanceId(a_MeshAndInstanceId)
        {}

        CPU_GPU INLINE bool IsIntersection() const
        {
            return (m_IntersectionT > 0.f);
        }

        //Read only
        unsigned int m_RayIndex;
        float m_IntersectionT;
        unsigned int m_TriangleId;
        unsigned int m_MeshAndInstanceId; //Might need to change to pointer to a DeviceMesh.

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

        CPU_GPU INLINE unsigned int GetSize() const
        {
            return m_NumPixels * m_IntersectionsPerPixel;
        }

        GPU_ONLY INLINE void SetIntersection(
            const IntersectionData& a_Data, 
            unsigned int a_PixelIndex, 
            unsigned int a_IntersectionIndex)
        {

            m_Intersections[GetRayIndex(a_PixelIndex, a_IntersectionIndex)] = a_Data;

        }

        GPU_ONLY INLINE const IntersectionData& GetIntersection(unsigned int a_PixelIndex, unsigned int a_IntersectionIndex)
        {

            return m_Intersections[GetRayIndex(a_PixelIndex, a_IntersectionIndex)];

        }

        GPU_ONLY INLINE const IntersectionData& GetIntersection(unsigned int a_RayIndex) const
        {
            assert(a_RayIndex < GetSize());

            return m_Intersections[a_RayIndex];
                
        }

        //Gets a index to IntersectionData in the m_Intersections array, taking into account the number of pixels and the number of rays per pixel.
        GPU_ONLY INLINE unsigned int GetRayIndex(unsigned int a_PixelIndex, unsigned int a_IntersectionIndex) const
        {

            assert(a_PixelIndex < m_NumPixels && a_IntersectionIndex < m_IntersectionsPerPixel);

            return a_PixelIndex * m_IntersectionsPerPixel + a_IntersectionIndex;

        }

    };

    struct PixelBuffer
    {

        PixelBuffer()
            :
        m_NumPixels(0u),
        m_ChannelsPerPixel(0u),
        m_Pixels()
        {}

        //Ready only
        const unsigned int m_NumPixels;
        const unsigned int m_ChannelsPerPixel;

        //Read/Write
        float3 m_Pixels[];

        GPU_ONLY INLINE void SetPixel(const float3& a_value, unsigned int a_PixelIndex, unsigned int a_ChannelIndex)
        {

            m_Pixels[GetPixelIndex(a_PixelIndex, a_ChannelIndex)] = a_value;

        }

        GPU_ONLY INLINE const float3& GetPixel(unsigned int a_PixelIndex, unsigned int a_ChannelIndex) const
        {

            return m_Pixels[GetPixelIndex(a_PixelIndex, a_ChannelIndex)];

        }

        //Gets an index to a pixel in the m_Pixels array, taking into account number of channels per pixel.
        GPU_ONLY INLINE unsigned int GetPixelIndex(unsigned int a_PixelIndex, unsigned int a_ChannelIndex) const
        {

            assert(a_PixelIndex < m_NumPixels&& a_ChannelIndex < m_ChannelsPerPixel);

            return a_PixelIndex * m_ChannelsPerPixel + a_ChannelIndex;

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
        m_PixelOutput(nullptr)
        {}

        //Read only (Might not be necessary to store the primary rays and intersections here)
        //const RayBatch* const m_PrimaryRays;
        //const IntersectionBuffer* const m_PrimaryIntersections;
        constexpr static unsigned int s_NumOutputChannels = static_cast<unsigned>(OutputChannel::NUM_CHANNELS);

        //Read/Write
        PixelBuffer* const m_PixelOutput;

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

        CPU_GPU INLINE unsigned int GetSize() const
        {
            return m_MaxDepth * m_NumPixels * m_RaysPerPixel;
        }

        GPU_ONLY INLINE void SetShadowRay(
            const ShadowRayData& a_Data, 
            unsigned int a_DepthIndex, 
            unsigned int a_PixelIndex, 
            unsigned int a_RayIndex = 0)
        {
            m_ShadowRays[GetShadowRayIndex(a_DepthIndex, a_PixelIndex, a_RayIndex)] = a_Data;
        }

        GPU_ONLY INLINE const ShadowRayData& GetShadowRayData(unsigned int a_DepthIndex, unsigned int a_PixelIndex, unsigned int a_RayIndex) const
        {

            return m_ShadowRays[GetShadowRayIndex(a_DepthIndex, a_PixelIndex, a_RayIndex)];

        }

        //Gets a index to a ShadowRay in the m_ShadowRays array, taking into account the max dept, number of pixels and number of rays per pixel.
        GPU_ONLY INLINE unsigned int GetShadowRayIndex(unsigned int a_DepthIndex, unsigned int a_PixelIndex, unsigned int a_RayIndex) const
        {

            assert(a_DepthIndex < m_MaxDepth&& a_PixelIndex < m_NumPixels&& a_RayIndex < m_RaysPerPixel);

            return a_DepthIndex * m_NumPixels * m_RaysPerPixel + a_PixelIndex * m_RaysPerPixel + a_RayIndex;

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

        unsigned int m_PixelIndex;
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
            const RayBatch* const a_PrimaryRaysPrevFrame,
            const IntersectionBuffer* a_PrimaryIntersectionsPrevFrame,
            const RayBatch* const a_CurrentRays,
            const IntersectionBuffer* a_CurrentIntersections,
            RayBatch* a_SecondaryRays,
            ShadowRayBatch* a_ShadowRayBatch,
            const LightBuffer* a_Lights)
            :
        m_ResolutionAndDepth(a_ResolutionAndDepth),
        m_PrimaryRaysPrevFrame(a_PrimaryRaysPrevFrame),
        m_PrimaryIntersectionsPrevFrame(a_PrimaryIntersectionsPrevFrame),
        m_CurrentRays(a_CurrentRays),
        m_CurrentIntersections(a_CurrentIntersections),
        m_LightBuffer(a_Lights),
        m_SecondaryRays(a_SecondaryRays),
        m_ShadowRaysBatch(a_ShadowRayBatch)
        {}

        CPU_ONLY ~ShadingLaunchParameters() = default;

        //Read only
        const uint3 m_ResolutionAndDepth;
        const RayBatch* const m_PrimaryRaysPrevFrame;
        const IntersectionBuffer* const m_PrimaryIntersectionsPrevFrame;
        const RayBatch* const m_CurrentRays;
        const IntersectionBuffer* const m_CurrentIntersections;
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
            const uint2& a_UpscaledResolution,
            const ResultBuffer* const a_WavefrontOutput,
            PixelBuffer* const a_MergedResults,
            uchar4* const a_ImageOutput)
            :
        m_Resolution(a_Resolution),
        m_UpscaledResolution(a_UpscaledResolution),
        m_WavefrontOutput(a_WavefrontOutput),
        m_MergedResults(a_MergedResults),
        m_ImageOutput(a_ImageOutput)
        {}

        CPU_ONLY ~PostProcessLaunchParameters() = default;

        //Read only
        const uint2 m_Resolution;
        const uint2 m_UpscaledResolution;
        const ResultBuffer* const m_WavefrontOutput;

        //Read/Write
        PixelBuffer* const m_MergedResults; //Used to merge results from multiple channels into one channel.
        uchar4* const m_ImageOutput; //Used to display image after DLSS algorithm has run on merged results.

    };



    // Shader Launch parameters

    struct CommonOptixLaunchParameters
    {

        uint3 m_ResolutionAndDepth;
        OptixTraversableHandle m_Traversable;

    };

    struct ResolveRaysLaunchParameters
    {

        CommonOptixLaunchParameters m_Common;

        RayBatch* m_Rays;
        IntersectionBuffer* m_Intersections;

    };

    struct ResolveShadowRaysLaunchParameters
    {

        CommonOptixLaunchParameters m_Common;

        ShadowRayBatch* m_ShadowRays;
        ResultBuffer* m_Results;

    };

    struct ResolveRaysRayGenData
    {
    };

    struct ResolveRaysHitData
    {
    };

    struct ResolveRaysMissData
    {
    };

    struct ResolveShadowRaysRayGenData
    {
    };

    struct ResolveShadowRaysHitData
    {
    };

    struct ResolveShadowRaysMissData
    {
    };

}

#undef CPU_ONLY 
#undef CPU_GPU
#undef GPU_ONLY 