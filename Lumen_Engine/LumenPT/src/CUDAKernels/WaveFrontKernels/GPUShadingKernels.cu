#include "GPUShadingKernels.cuh"
#include <device_launch_parameters.h>

//Just temporary CUDA kernels.

CPU_ON_GPU void DEBUGShadePrimIntersections(
    const uint3 a_ResolutionAndDepth,
    const SurfaceData* a_CurrentSurfaceData,
    float3* const a_Output
)
{

    const unsigned int numPixels = a_ResolutionAndDepth.x * a_ResolutionAndDepth.y;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < numPixels; i += stride)
    {
        //Copy the diffuse color over for now for unshaded.
        auto* output = &a_Output[i * static_cast<unsigned>(LightChannel::NUM_CHANNELS)];
        output[static_cast<unsigned>(LightChannel::DIRECT)] = a_CurrentSurfaceData[i].m_Color;
    }
}


CPU_ON_GPU void WriteToOutput(
    const uint2 a_Resolution,
    const float3 * const a_Input,
    uchar4* a_Output
)
{
    const unsigned int numPixels = a_Resolution.x * a_Resolution.y;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    //This literally copies 1-to-1 so it doesn't need to know about pixel indices or anything.
    //TODO: Maybe skip this step entirely and just directly output to this buffer when merging light channels? Then apply effects in this buffer?
    //TODO: It would save one copy.
    for (unsigned int i = index; i < numPixels; i += stride)
    {
        a_Output[i] = make_color(a_Input[i]);
    }
}

CPU_ON_GPU void GenerateMotionVector(MotionVectorBuffer* a_Buffer, 
const SurfaceData* a_CurrentSurfaceData, 
uint2 a_Resolution, 
sutil::Matrix4x4 a_PrevViewProjMatrix
)
{
    int size = a_Resolution.x * a_Resolution.y;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < size; i += stride)
    {
		float4 currWorldPos = make_float4(a_CurrentSurfaceData[i].m_Position, 1.0f);
        float2 currScreenPos;
    	currScreenPos.y = (static_cast<unsigned int>(i) / a_Resolution.x);
    	currScreenPos.x = (static_cast<unsigned int>(i) - currScreenPos.y * a_Resolution.x);
    	currScreenPos += make_float2(0.5f);
        currScreenPos.x /= static_cast<float>(a_Resolution.x);
        currScreenPos.y /= static_cast<float>(a_Resolution.y);
    	
    	float4 prevClipCoordinates = a_PrevViewProjMatrix * currWorldPos;
        float3 prevNdc = make_float3(prevClipCoordinates.x, prevClipCoordinates.y, prevClipCoordinates.z) / prevClipCoordinates.w;
    	float2 prevScreenPos = make_float2(
        (prevNdc.x * 0.5f + 0.5f)/* * static_cast<float>(a_Resolution.x)*/,
        (prevNdc.y * 0.5f + 0.5f)/* * static_cast<float>(a_Resolution.y)*/
        );

        MotionVectorData motionVectorData;
        motionVectorData.m_Velocity = (currScreenPos - prevScreenPos);
        
        a_Buffer->SetMotionVectorData(motionVectorData, i);
    	
        //if(i == 800 * 300 + 400)
        {
	        //printf("x: %.6f y: %.6f \n", a_Buffer->m_MotionVectorBuffer[i].m_Velocity.x, a_Buffer->m_MotionVectorBuffer[i].m_Velocity.y);
            //printf("x: %.6f y: %.6f \n", currScreenPos.x, currScreenPos.y);
            //printf("x: %.6f y: %.6f \n", prevScreenPos.x, prevScreenPos.y);
        }
    }
}