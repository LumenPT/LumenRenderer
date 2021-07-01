#include "GPUShadingKernels.cuh"
#include "../../Shaders/CppCommon/Half4.h"

#include <device_launch_parameters.h>

CPU_ON_GPU void ExtractDepthDataGpu(
    const SurfaceData* a_SurfaceData,
    cudaSurfaceObject_t a_DepthOutPut,
    uint2 a_Resolution,
    float2 a_MinMaxDistance)
{

    const unsigned int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    const unsigned int pixelDataIndex = PIXEL_DATA_INDEX(pixelX, pixelY, a_Resolution.x);

    //Check if a_OutPut->m_IntersectionT > a_DepthOutPut->depthValueAtPixelIndex to avoid writing to it if the T < value already there
    //if T > valueAtPixel ? overwrite : keep valueAtPixel

    if (pixelX < a_Resolution.x && pixelY < a_Resolution.y)
    {
        float t = a_SurfaceData[pixelDataIndex].m_IntersectionT;

        //float1 t = make_float1(a_SurfaceData[pixelDataIndex].m_IntersectionT);

        if (a_SurfaceData[pixelDataIndex].m_IntersectionT < 0.f)  //below 0 == no intersection
        {
            float1 nullResult = make_float1(0.f);

            surf2Dwrite<float1>(
                nullResult,
                a_DepthOutPut,  //intput
                pixelX * sizeof(float1),
                pixelY,
                cudaBoundaryModeTrap);

            //half4Ushort4 nullResult = { make_float4(0.f, 0.f, 0.f, 0.f) };
            //surf2Dwrite<ushort4>(
            //    nullResult.m_Ushort4,
            //    a_DepthOutPut,  //intput
            //    pixelX * sizeof(ushort4),
            //    pixelY,
            //    cudaBoundaryModeTrap);
            return;
        }

        t = (t - fminf(a_MinMaxDistance.x, t)) / (fmaxf(a_MinMaxDistance.y, t) - fminf(a_MinMaxDistance.x, t));

        //float tt = (t - fminf(0.f, t)) / (fmaxf(1.f, t) - fminf(0.f, t));
        //result = { make_float4(0, 0, 0, 0) };


        //half4Ushort4 result = { make_float4(t, t, t, t) };
        //surf2Dwrite<ushort4>(
        //    result.m_Ushort4,
        //    a_DepthOutPut,  //intput
        //    pixelX * sizeof(ushort4),
        //    pixelY,
        //    cudaBoundaryModeTrap);

        float1 result = make_float1(t);
        surf2Dwrite<float1>(
            result,
            a_DepthOutPut,  //intput
            pixelX * sizeof(float1),
            pixelY,
            cudaBoundaryModeTrap);

    }

}