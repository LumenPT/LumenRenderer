#include "ReSTIRKernels.cuh"

__host__ void ResetReservoirs(int a_NumReservoirs, Reservoir* a_ReservoirPointer)
{
    //Call in parallel.
    const int blockSize = 256;
    const int numBlocks = (a_NumReservoirs + blockSize - 1) / blockSize;
    ResetReservoirInternal<<<numBlocks, blockSize>>>(a_NumReservoirs, a_ReservoirPointer);
    cudaDeviceSynchronize();
}

__global__ void ResetReservoirInternal(int a_NumReservoirs, Reservoir* a_ReservoirPointer)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < a_NumReservoirs; i += stride)
    {
        a_ReservoirPointer[i].Reset();
    }
}

__host__ void FillCDF(CDF* a_Cdf, TriangleLight* a_Lights, unsigned a_LightCount)
{
    //First reset the CDF on the GPU.
    ResetCDF<<<1,1>>>(a_Cdf);

    //Run from one thread because it's not thread safe to append the sum of each element.
    FillCDFInternal <<<1, 1>>> (a_Cdf, a_Lights, a_LightCount);
    cudaDeviceSynchronize();
}

__global__ void ResetCDF(CDF* a_Cdf)
{
    a_Cdf->Reset();
}

__global__ void FillCDFInternal(CDF* a_Cdf, TriangleLight* a_Lights, unsigned a_LightCount)
{
    for (int i = 0; i < a_LightCount; ++i)
    {
        //Weight is the average illumination for now. Could take camera into account.
        const float3 radiance = a_Lights[i].radiance;
        a_Cdf->Insert((radiance.x + radiance.y + radiance.z) / 3.f);
    }
}

__host__ void FillLightBags(unsigned a_NumLightBags, CDF* a_Cdf, LightBagEntry* a_LightBagPtr)
{
    const int blockSize = 256;
    const int numBlocks = (a_NumLightBags + blockSize - 1) / blockSize;
    FillLightBagsInternal << <numBlocks, blockSize >> > (a_NumLightBags, a_Cdf, a_LightBagPtr);
    cudaDeviceSynchronize();
}

__global__ void FillLightBagsInternal(unsigned a_NumLightBags, CDF* a_Cdf, LightBagEntry* a_LightBagPtr)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < a_NumLightBags; i += stride)
    {
        //TODO generate random float between 0 and 1.
        float random = 1.f;

        //Store the pdf and light in the light bag.
        unsigned lIndex;
        float pdf;
        a_Cdf->Get(random, lIndex, pdf);
        a_LightBagPtr[i] = {lIndex, pdf};
    }
}
