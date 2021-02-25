#pragma once
#include "CudaDefines.h"
#include "ModelStructs.h"
#include "ReSTIRData.h"

#include <Optix/optix.h>
#include <Cuda/cuda/helpers.h>
#include <cassert>
#include <cstdio>
#include <array>


/*TODO: check if usage of textures can benefit performance
 *for buffers like intersectionBuffer or OutputBuffer.
*/

namespace WaveFront
{

    template<typename T>
    CPU_GPU bool IsNotNaN(T a_Val)
    {
        return (a_Val == a_Val);
    }

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


    /// <summary>
    /// <b>Description</b> \n
    /// Stores definition of a ray and additional data.\n
    /// <b>Type</b>: Struct\n
    /// <para>
    /// <b>Member variables:</b> \n
    /// <b>• m_Origin</b> <em>(float3)</em>: Origin of the ray. \n
    /// <b>• m_Direction</b> <em>(float3)</em>: Direction of the ray. \n
    /// <b>• m_Contribution</b> <em>(float3)</em>: Contribution scalar for the returned radiance. \n
    /// </para>
    /// </summary>
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



        /// <summary> Checks if the ray is a valid ray.</summary>
        /// <returns> Returns true if <b>all</b> of the components of m_Direction are not equal to 0. <em>(boolean)</em> </returns>
        GPU_ONLY INLINE bool IsValidRay() const
        {

            return (m_Direction.x != 0.f ||
                    m_Direction.y != 0.f ||
                    m_Direction.z != 0.f);

        }



        /// <summary>
        /// <b>Description</b> \n Stores the position of the ray interpreted as a world-space position. \n
        /// <b>Default</b>: (0.f, 0.f, 0.f)
        /// </summary>
        float3 m_Origin;
        /// <summary>
        /// <b>Description</b> \n Stores the direction of the ray interpreted as a normalized vector. \n
        /// <b>Default</b>: (0.f, 0.f, 0.f)
        /// </summary>
        float3 m_Direction;
        /// <summary>
        /// <b>Description</b> \n Stores the contribution of the radiance returned by the ray as a scalar for each rgb-channel. \n
        /// <b>Default</b>: (0.f, 0.f, 0.f)
        /// </summary>
        float3 m_Contribution;

    };

    /// <summary>
    /// <b>Description</b> \n
    /// Stores a buffer of RayData structs. \n
    /// <b>Type</b>: Struct\n
    /// <para>
    /// <b>Member variables</b> \n
    /// <b>• m_NumPixels</b> <em>(const unsigned int)</em>: Number of pixels the buffer stores data for.\n
    /// <b>• m_RaysPerPixel</b> <em>(const unsigned int)</em>: Number of rays per pixel the buffer stores data for.\n
    /// <b>• m_Rays</b> <em>(RayData[])</em>: Array storing all the RayData structs.\n
    /// </para>
    /// </summary>
    struct RayBatch
    {

        RayBatch()
            :
            m_NumPixels(0u),
            m_RaysPerPixel(0u),
            m_Rays()
        {}



        /// <summary> Gets the size of the buffer. \n Takes into account the number of pixels and the number of rays per pixel. </summary>
        /// <returns> Size of the buffer.  <em>(unsigned int)</em> </returns>
        CPU_GPU INLINE unsigned int GetSize() const
        {
            return m_NumPixels * m_RaysPerPixel;
        }

        /// <summary> Sets the ray data from a ray for a certain sample for a certain pixel. </summary>
        /// <param name="a_Data">\n • Description: Ray data to set.</param>
        /// <param name="a_PixelIndex">\n • Description: Index of the pixel to set the ray data for. \n • Range: (0 : m_NumPixels-1) </param>
        /// <param name="a_RayIndex">\n • Description: Index of the sample to set the ray data for. \n • Range: (0 : m_RaysPerPixel-1)</param>
        GPU_ONLY INLINE void SetRay(
            const RayData& a_Data, 
            unsigned int a_PixelIndex,
            unsigned int a_RayIndex = 0)
        {
            m_Rays[GetRayArrayIndex(a_PixelIndex, a_RayIndex)] = a_Data;
        }

        /// <summary> Gets the ray data from a ray for a certain sample for a certain pixel. </summary>
        /// <param name="a_PixelIndex">\n • Description: The index of the pixel to get the ray data from. \n • Range (0 : m_NumPixels-1)</param>
        /// <param name="a_RayIndex">\n • Description: The index of the ray for the pixel. \n • Range: (0 : m_RaysPerPixel-1)</param>
        /// <returns> Data of the specified ray. <em>(const RayData&)</em>  </returns>
        GPU_ONLY INLINE const RayData& GetRay(unsigned int a_PixelIndex, const unsigned int a_RayIndex) const
        {

            return m_Rays[GetRayArrayIndex(a_PixelIndex, a_RayIndex)];

        }

        /// <summary Gets the ray data from a ray for a certain sample for a certain pixel </summary>
        /// <remarks To get the right index for the pixel and sample you can use the GetRayArrayIndex function </remarks>
        /// <param name="a_RayArrayIndex">\n • Description: The index in the m_Rays array to get the ray data from. \n • Range (0 : (m_NumPixels * m_RaysPerPixel)-1)</param>
        /// <returns> Data of the specified ray. <em>(const RayData&)</em>  </returns>
        GPU_ONLY INLINE const RayData& GetRay(unsigned int a_RayArrayIndex) const
        {
            assert(a_RayArrayIndex < GetSize());

            return m_Rays[a_RayArrayIndex];

        }

        /// <summary> Get the index in the m_Rays array for a certain sample at a certain pixel </summary>
        /// <param name="a_PixelIndex">\n • Description: The index of the pixel to get the index for. \n • Range (0 : m_NumPixels-1)</param>
        /// <param name="a_RayIndex">\n • Description: The index of the sample to get the index for. \n • Range (0 : m_RaysPerPixel-1)</param>
        /// <returns> Index into the m_Rays array for the sample at the pixel. <em>(unsigned int)</em>  </returns>
        GPU_ONLY INLINE unsigned int GetRayArrayIndex(unsigned int a_PixelIndex, const unsigned int a_RayIndex = 0) const
        {

            assert(a_PixelIndex < m_NumPixels&& a_RayIndex < m_RaysPerPixel);

            return a_PixelIndex * m_RaysPerPixel + a_RayIndex;

        }



        /// <summary>
        /// <b>Description</b> \n The number of pixels the buffer stores data for. \n
        /// <b>Default</b>: 0
        /// </summary>
        const unsigned int m_NumPixels;

        /// <summary>
        /// <b>Description</b> \n  The number of rays per pixel the buffer stores data for. \n
        /// <b>Default</b>: 0
        /// </summary>
        const unsigned int m_RaysPerPixel;

        /// <summary>
        /// <b>Description</b> \n Array storing the RayData structs. \n Has a size of m_NumPixels * m_RaysPerPixel. \n
        /// <b>Default</b>: empty
        /// </summary>
        RayData m_Rays[];

    };

    /// <summary>
    /// <b>Description</b> \n
    /// Stores data of an intersection. \n
    /// <b>Type</b>: Struct\n
    /// <para>
    /// <b>Member variables</b> \n
    /// <b>• m_RayIndex</b> <em>(unsigned int)</em>: Index in the m_Rays member of a RayBatch of the ray the intersection belongs to.\n
    /// <b>• m_IntersectionT</b> <em>(float)</em>: Distance along the ray to the intersection.\n
    /// <b>• m_PrimitiveIndex</b> <em>(unsigned int): Index of the primitive of the mesh intersected by the ray.</em>: .\n
    /// </para>
    /// </summary>
    struct IntersectionData
    {

        CPU_GPU IntersectionData()
            :
        m_RayArrayIndex(0),
        m_IntersectionT(-1.f),
        m_PrimitiveIndex(0),
        m_Primitive(0)
        {}

        CPU_GPU IntersectionData(
            unsigned int a_RayArrayIndex,
            float a_IntersectionT,
            unsigned int a_PrimitiveIndex,
            DevicePrimitive* a_Primitive)
            :
        m_RayArrayIndex(a_RayArrayIndex),
        m_IntersectionT(a_IntersectionT),
        m_PrimitiveIndex(a_PrimitiveIndex),
        m_Primitive(a_Primitive)
        {}



        /// <summary> Checks if the data defines an intersection. </summary>
        /// <returns> Returns true if m_IntersectionT is higher than 0.  <em>(boolean)</em> </returns>
        CPU_GPU INLINE bool IsIntersection() const
        {
            return (m_IntersectionT > 0.f);
        }



        /// <summary>
        /// <b>Description</b> \n The index in the m_Rays array of a RayBatch of the ray the intersection belongs to. \n
        /// <b>Default</b>: 0
        /// </summary>
        unsigned int m_RayArrayIndex;

        /// <summary>
        /// <b>Description</b> \n Distance along the ray the intersection happened. \n
        /// <b>Default</b>: -1.f
        /// </summary>
        float m_IntersectionT;

        /// <summary>
        /// <b>Description</b> \n The index of the primitive of the mesh that the ray intersected with. \n
        /// <b>Default</b>: 0
        /// </summary>
        unsigned int m_PrimitiveIndex;

        /// <summary>
        /// <b>Description</b> \n The index of the primitive of the mesh that the ray intersected with. \n
        /// <b>Default</b>: 0
        /// </summary>
        DevicePrimitive* m_Primitive; //TODO: Might need to change to pointer to a DeviceMesh.

    };

    struct IntersectionBuffer
    {

        IntersectionBuffer()
            :
        m_NumPixels(0u),
        m_IntersectionsPerPixel(0u),
        m_Intersections()
        {}



        CPU_GPU INLINE unsigned int GetSize() const
        {
            return m_NumPixels * m_IntersectionsPerPixel;
        }

        GPU_ONLY INLINE void SetIntersection(
            const IntersectionData& a_Data, 
            unsigned int a_PixelIndex, 
            unsigned int a_IntersectionIndex = 0)
        {

            m_Intersections[GetIntersectionArrayIndex(a_PixelIndex, a_IntersectionIndex)] = a_Data;

        }

        GPU_ONLY INLINE const IntersectionData& GetIntersection(unsigned int a_PixelIndex, unsigned int a_IntersectionIndex) const
        {

            return m_Intersections[GetIntersectionArrayIndex(a_PixelIndex, a_IntersectionIndex)];

        }

        GPU_ONLY INLINE const IntersectionData& GetIntersection(unsigned int a_RayIndex) const
        {
            assert(a_RayIndex < GetSize());

            return m_Intersections[a_RayIndex];
                
        }

        //Gets a index to IntersectionData in the m_Intersections array, taking into account the number of pixels and the number of rays per pixel.
        GPU_ONLY INLINE unsigned int GetIntersectionArrayIndex(unsigned int a_PixelIndex, unsigned int a_IntersectionIndex = 0) const
        {

            assert(a_PixelIndex < m_NumPixels && a_IntersectionIndex < m_IntersectionsPerPixel);

            return a_PixelIndex * m_IntersectionsPerPixel + a_IntersectionIndex;

        }



        //Read only
        const unsigned int m_NumPixels;
        const unsigned int m_IntersectionsPerPixel;

        //Read/Write
        IntersectionData m_Intersections[];

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

        CPU_GPU unsigned int GetSize() const
        {
            return m_NumPixels * m_ChannelsPerPixel;
        }

        //Gets an index to a pixel in the m_Pixels array, taking into account number of channels per pixel.
        GPU_ONLY INLINE unsigned int GetPixelArrayIndex(unsigned int a_PixelIndex, unsigned int a_ChannelIndex = 0) const
        {

            assert(a_PixelIndex < m_NumPixels&& a_ChannelIndex < m_ChannelsPerPixel);

            return a_PixelIndex * m_ChannelsPerPixel + a_ChannelIndex;

        }

        GPU_ONLY INLINE const float3& GetPixel(unsigned int a_PixelIndex, unsigned int a_ChannelIndex) const
        {

            return m_Pixels[GetPixelArrayIndex(a_PixelIndex, a_ChannelIndex)];

        }

        GPU_ONLY INLINE const float3& GetPixel(unsigned int a_PixelArrayIndex) const
        {

            assert(a_PixelArrayIndex < GetSize());

            return m_Pixels[a_PixelArrayIndex];

        }

        GPU_ONLY INLINE void SetPixel(float3 a_value, unsigned int a_PixelIndex, unsigned int a_ChannelIndex)
        {

            /*if(a_value.x != 0.f || a_value.y != 0.f || a_value.z != 0.f)
            {
                printf("PixelBuffer: SetPixel: %f, %f, %f\n", a_value.x, a_value.y, a_value.z);
            }*/

            float3& pixel = m_Pixels[GetPixelArrayIndex(a_PixelIndex, a_ChannelIndex)];
            pixel = a_value;

            /*if (a_value.x != 0.f || a_value.y != 0.f || a_value.z != 0.f)
            {

                float3 result = m_Pixels[GetPixelArrayIndex(a_PixelIndex, a_ChannelIndex)];
                printf("PixelBuffer: SetPixel Result: %f, %f, %f \n", result.x, result.y, result.z);

            }*/

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

        constexpr static unsigned int s_NumOutputChannels = static_cast<unsigned>(OutputChannel::NUM_CHANNELS);



        ResultBuffer()
            :
        m_PixelBuffer(nullptr)
        {}



        CPU_GPU static unsigned int GetNumOutputChannels()
        {
            return static_cast<unsigned>(OutputChannel::NUM_CHANNELS);
        }

        GPU_ONLY INLINE void SetPixel(float3 a_Value, unsigned a_PixelIndex, OutputChannel a_Channel)
        {

            assert(a_Channel != OutputChannel::NUM_CHANNELS);

            m_PixelBuffer->SetPixel(a_Value, a_PixelIndex, static_cast<unsigned>(a_Channel));

        }

        GPU_ONLY INLINE void SetPixel(const float3 a_Values[static_cast<unsigned>(OutputChannel::NUM_CHANNELS)], unsigned a_PixelIndex)
        {

            const unsigned numOutputChannels = GetNumOutputChannels();
            for(unsigned int i = 0u; i < numOutputChannels; ++i)
            {
                m_PixelBuffer->SetPixel(a_Values[i], a_PixelIndex, i);
            }

        }

        GPU_ONLY INLINE const float3& GetPixel(unsigned a_PixelIndex, OutputChannel a_channel) const
        {

            assert(a_channel != OutputChannel::NUM_CHANNELS);

            return m_PixelBuffer->GetPixel(a_PixelIndex, static_cast<unsigned>(a_channel));

        }

        GPU_ONLY float3 GetPixelCombined(unsigned a_PixelIndex) const
        {

            float3 result = make_float3(0.0f);

            const unsigned numOutputChannels = GetNumOutputChannels();

            for(unsigned i = 0; i < numOutputChannels; ++i)
            {

                const float3& color = m_PixelBuffer->GetPixel(a_PixelIndex, i);
                result += color;

            }

            return result;

        }



        PixelBuffer* const m_PixelBuffer;

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

        GPU_ONLY INLINE bool IsValidRay() const
        {

            return  (m_Direction.x != 0.f ||
                     m_Direction.y != 0.f ||
                     m_Direction.z != 0.f)&& 
                     m_MaxDistance > 0.f &&
                     IsNotNaN(m_Direction.x) &&
                     IsNotNaN(m_Direction.y) &&
                     IsNotNaN(m_Direction.z) &&
                     IsNotNaN(m_Origin.x) &&
                     IsNotNaN(m_Origin.y) &&
                     IsNotNaN(m_Origin.z) &&
                     IsNotNaN(m_MaxDistance);

        }

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
            m_ShadowRays[GetShadowRayArrayIndex(a_DepthIndex, a_PixelIndex, a_RayIndex)] = a_Data;
        }

        GPU_ONLY INLINE const ShadowRayData& GetShadowRayData(unsigned int a_DepthIndex, unsigned int a_PixelIndex, unsigned int a_RayIndex) const
        {

            return m_ShadowRays[GetShadowRayArrayIndex(a_DepthIndex, a_PixelIndex, a_RayIndex)];

        }

        GPU_ONLY INLINE const ShadowRayData& GetShadowRayData(unsigned int a_ShadowRayArrayIndex) const
        {

            assert(a_ShadowRayArrayIndex < GetSize());

            return m_ShadowRays[a_ShadowRayArrayIndex];

        }

        //Gets a index to a ShadowRay in the m_ShadowRays array, taking into account the max dept, number of pixels and number of rays per pixel.
        GPU_ONLY INLINE unsigned int GetShadowRayArrayIndex(unsigned int a_DepthIndex, unsigned int a_PixelIndex, unsigned int a_RayIndex) const
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
            const LightBuffer* a_Lights,
            CDF* const a_CDF = nullptr)
            :
        m_ResolutionAndDepth(a_ResolutionAndDepth),
        m_PrimaryRaysPrevFrame(a_PrimaryRaysPrevFrame),
        m_PrimaryIntersectionsPrevFrame(a_PrimaryIntersectionsPrevFrame),
        m_CurrentRays(a_CurrentRays),
        m_CurrentIntersections(a_CurrentIntersections),
        m_LightBuffer(a_Lights),
        m_SecondaryRays(a_SecondaryRays),
        m_ShadowRaysBatch(a_ShadowRayBatch),
		m_CDF(a_CDF)
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
        CDF* const m_CDF;
    	
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



    // OptiX shader data parameters
    // Launch parameters

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



    // Shader helper structs

    struct Occlusion
    {

        CPU_GPU Occlusion(float a_MaxDistance)
            :
            m_MaxDistance(a_MaxDistance),
            m_Occluded(false)
        {}

        const float m_MaxDistance;
        bool m_Occluded;

    };



    // ShaderBindingTable data

    struct ResolveRaysRayGenData
    {

        float m_MinDistance;
        float m_MaxDistance;

    };

    struct ResolveRaysHitData
    {

        int m_Dummy;

    };

    struct ResolveRaysMissData
    {

        int m_Dummy;

    };

    struct ResolveShadowRaysRayGenData
    {

        float m_MinDistance;
        float m_MaxDistance;

    };

    struct ResolveShadowRaysHitData
    {

        int m_Dummy;

    };

    struct ResolveShadowRaysMissData
    {

        int m_Dummy;

    };

}