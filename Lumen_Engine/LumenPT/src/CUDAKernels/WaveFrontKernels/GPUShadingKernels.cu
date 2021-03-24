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
    uchar4* a_Output)
{
    const unsigned int numPixels = a_Resolution.x * a_Resolution.y;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    //This literally copies 1-to-1 so it doesn't need to know about pixel indices or anything.
    //TODO: Maybe skip this step entirely and just directly output to this buffer when merging light channels? Then apply effects in this buffer?
    //TODO: It would save one copy.
    for (unsigned int i = index; i < numPixels; i += stride)
    {
        const auto color = make_color(a_Input[i]);
        a_Output[i] = color;
    }
}

CPU_ON_GPU void GenerateMotionVector(MotionVectorBuffer* a_Buffer, 
const SurfaceData* a_CurrentSurfaceData, 
uint2 a_Resolution, 
sutil::Matrix4x4 m_PrevViewMatrix,
sutil::Matrix4x4 m_ProjectionMatrix
)
{
    int size = a_Resolution.x * a_Resolution.y;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < size; i += stride)
    {
		float4 currWorldPos = make_float4(a_CurrentSurfaceData[i].m_Position, 1.0f);
        float2 currScreenCoord;
    	currScreenCoord.y = (static_cast<unsigned int>(i) / a_Resolution.x);
    	currScreenCoord.x = (static_cast<unsigned int>(i) - currScreenCoord.y * a_Resolution.x);
    	currScreenCoord += make_float2(0.5f);
    	
    	float4 prevClipCoordinates = m_ProjectionMatrix * m_PrevViewMatrix * currWorldPos;
        float3 prevNdc = make_float3(prevClipCoordinates.x, prevClipCoordinates.y, prevClipCoordinates.z) / prevClipCoordinates.w;
    	float2 prevScreenCoord = make_float2(
        (prevNdc.x * 0.5f + 0.5f) * static_cast<float>(a_Resolution.x),
        (prevNdc.y * 0.5f + 0.5f) * static_cast<float>(a_Resolution.y)
        );

        MotionVectorData motionVectorData;
        motionVectorData.m_Velocity = (currScreenCoord - prevScreenCoord);
        
        a_Buffer->SetMotionVectorData(motionVectorData, i);
    	
        a_Buffer->m_MotionVectorBuffer[0].m_Velocity;
        printf("x: %.6f y: %.6f \n", a_Buffer->m_MotionVectorBuffer[i].m_Velocity.x, a_Buffer->m_MotionVectorBuffer[i].m_Velocity.y);
        //printf("curr x: %.6f curr y: %.6f prev x: %.6f prev y: %.6f \n", currScreenCoord.x, currScreenCoord.y, prevScreenCoord.x, prevScreenCoord.y);
    	
    	//if(
        //(screenPos.x == 0 && screenPos.y == 0) ||
        //(screenPos.x == 400 && screenPos.y == 300) ||
    	//(screenPos.x == 799 && screenPos.y == 599) ||
    	//(screenPos.x == 355)
        //)
    	{
    		//printf("screen x: %d screen y: %d prev screen x: %.6f prev screen y: %.6f \n", screenPos.x, screenPos.y, screenCoordinates.x, screenCoordinates.y);
			//printf("screen x: %d screen y: %d prev screen x: %.6f prev screen y: %.6f \n", screenPos.x, screenPos.y, ndc.x, ndc.y);
    	}
    }
}