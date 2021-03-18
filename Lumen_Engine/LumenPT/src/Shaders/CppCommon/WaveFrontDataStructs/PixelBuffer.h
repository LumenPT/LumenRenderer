#pragma once
#include "../CudaDefines.h"

#include <Optix/optix.h>
#include <Cuda/cuda/helpers.h>
#include <cassert>

namespace WaveFront
{

    struct PixelBuffer
    {

        PixelBuffer()
            :
            m_NumPixels(0u),
            m_ChannelsPerPixel(0u),
            m_Pixels()
        {}



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

            float3& pixel = m_Pixels[GetPixelArrayIndex(a_PixelIndex, a_ChannelIndex)];
            pixel = a_value;

        }



        //Ready only
        const unsigned int m_NumPixels;
        const unsigned int m_ChannelsPerPixel;

        //Read/Write
        float3 m_Pixels[];

    };

}
