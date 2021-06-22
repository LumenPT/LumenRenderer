#include "GPUShadingKernels.cuh"
#include "../../Shaders/CppCommon/Half4.h"

#include <device_launch_parameters.h>

CPU_ON_GPU void ExtractDepthDataGpu(
    const SurfaceData* a_SurfaceData, 
    cudaSurfaceObject_t a_DepthOutPut,
    uint2 a_Resolution)
{

    const unsigned int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    const unsigned int pixelDataIndex = PIXEL_DATA_INDEX(pixelX, pixelY, a_Resolution.x);

    //Check if a_OutPut->m_IntersectionT > a_DepthOutPut->depthValueAtPixelIndex to avoid writing to it if the T < value already there
    //if T > valueAtPixel ? overwrite : keep valueAtPixel
    
    if (pixelX < a_Resolution.x && pixelY < a_Resolution.y)
    {
        if (a_SurfaceData[pixelDataIndex].m_IntersectionT < 0)
        {
            return; 
        }

        float t = a_SurfaceData[pixelDataIndex].m_IntersectionT;

        float tt = (t - fminf(0.f, t)) / (fmaxf(1.f, t) - fminf(0.f, t));

        half4Ushort4 result = { make_float4(tt, tt, tt, tt) };

        surf2Dwrite<ushort4>(
            result.m_Ushort4,
            a_DepthOutPut,  //intput
            pixelX * sizeof(ushort4),
            pixelY,
            cudaBoundaryModeTrap);

    }

}