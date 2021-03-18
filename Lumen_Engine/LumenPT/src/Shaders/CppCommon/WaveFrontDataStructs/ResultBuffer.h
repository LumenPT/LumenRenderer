#pragma once
#include "../CudaDefines.h"
#include "PixelBuffer.h"

#include <Optix/optix.h>
#include <Cuda/cuda/helpers.h>
#include <cassert>

namespace WaveFront
{

    struct ResultBuffer
    {

        enum class OutputChannel : unsigned int
        {
            DIRECT,
            INDIRECT,
            SPECULAR,
            NUM_CHANNELS
        };

        constexpr static unsigned int s_NumOutputChannels = static_cast<unsigned int>(OutputChannel::NUM_CHANNELS);



        ResultBuffer()
            :
            m_PixelBuffer(nullptr)
        {}



        CPU_GPU static unsigned int GetNumOutputChannels()
        {
            return static_cast<unsigned int>(OutputChannel::NUM_CHANNELS);
        }

        GPU_ONLY INLINE void SetPixel(float3 a_Value, unsigned int a_PixelIndex, OutputChannel a_Channel)
        {

            assert(a_Channel != OutputChannel::NUM_CHANNELS);

            m_PixelBuffer->SetPixel(a_Value, a_PixelIndex, static_cast<unsigned int>(a_Channel));

        }

        GPU_ONLY INLINE void SetPixel(float3 a_Values[s_NumOutputChannels], unsigned int a_PixelIndex)
        {

            const unsigned numOutputChannels = GetNumOutputChannels();
            for (unsigned int i = 0u; i < numOutputChannels; ++i)
            {
                m_PixelBuffer->SetPixel(a_Values[i], a_PixelIndex, i);
            }

        }

        GPU_ONLY INLINE const float3& GetPixel(unsigned int a_PixelIndex, OutputChannel a_channel) const
        {

            assert(a_channel != OutputChannel::NUM_CHANNELS);

            return m_PixelBuffer->GetPixel(a_PixelIndex, static_cast<unsigned int>(a_channel));

        }

        GPU_ONLY float3 GetPixelCombined(unsigned int a_PixelIndex) const
        {

            float3 result = make_float3(0.0f);

            const unsigned int numOutputChannels = GetNumOutputChannels();

            for (unsigned int i = 0; i < numOutputChannels; ++i)
            {

                const float3& color = m_PixelBuffer->GetPixel(a_PixelIndex, i);
                result += color;

            }

            return result;

        }



        PixelBuffer* const m_PixelBuffer;

    };

}
