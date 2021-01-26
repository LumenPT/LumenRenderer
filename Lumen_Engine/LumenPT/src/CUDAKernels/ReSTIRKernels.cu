#include "ReSTIRKernels.cuh"

#include "../Shaders/CppCommon/WaveFrontDataStructs.h"

__host__ void ResetReservoirs(int a_NumReservoirs, Reservoir* a_ReservoirPointer)
{
    //Call in parallel.
    const int blockSize = 256;
    const int numBlocks = (a_NumReservoirs + blockSize - 1) / blockSize;
    ResetReservoirInternal<<<numBlocks, blockSize>>>(a_NumReservoirs, a_ReservoirPointer);

    //TODO: Wait after every task may not be needed.Check if it is required between kernel calls.
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
    //TODO: This is not efficient single threaded.
    //TODO: Use this: https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
    //TODO: Interpret the array as a binary tree, then sum parts of it in sweeps. Then combine after. 

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

__host__ void FillLightBags(unsigned a_NumLightBags, CDF* a_Cdf, LightBagEntry* a_LightBagPtr, TriangleLight* a_Lights)
{
    const int blockSize = 256;
    const int numBlocks = (a_NumLightBags + blockSize - 1) / blockSize;
    FillLightBagsInternal <<<numBlocks, blockSize >>>(a_NumLightBags, a_Cdf, a_LightBagPtr, a_Lights);
    cudaDeviceSynchronize();
}

__global__ void FillLightBagsInternal(unsigned a_NumLightBags, CDF* a_Cdf, LightBagEntry* a_LightBagPtr, TriangleLight* a_Lights)
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
        a_LightBagPtr[i] = {a_Lights[lIndex], pdf};
    }
}

__host__ void PickPrimarySamples(const WaveFront::RayData* const a_RayData, const WaveFront::IntersectionData* const a_IntersectionData, const LightBagEntry* const a_LightBags, Reservoir* a_Reservoirs, const ReSTIRSettings& a_Settings)
{
    //TODO ensure that each pixel grid operates within a single block, and that the L1 cache is not overwritten for each value. Optimize for cache hits.
    //TODO correctly assign a light bag per grid through some random generation.

    const auto numReservoirs = (a_Settings.width * a_Settings.height * a_Settings.numReservoirsPerPixel);
    const int blockSize = 256;
    const int numBlocks = (numReservoirs + blockSize - 1) / blockSize;
    PickPrimarySamplesInternal<<<numBlocks, blockSize>>>(a_RayData, a_IntersectionData, a_LightBags, a_Reservoirs, a_Settings.numPrimarySamples, numReservoirs, a_Settings.numLightsPerBag, a_Settings.numLightBags);
    cudaDeviceSynchronize();
}

__global__ void PickPrimarySamplesInternal(const WaveFront::RayData* const a_RayData, const WaveFront::IntersectionData* const a_IntersectionData, const LightBagEntry* const a_LightBags, Reservoir* a_Reservoirs, const unsigned a_NumPrimarySamples, const unsigned a_NumReservoirs, const unsigned a_NumLightsPerBag, const unsigned a_NumLightBags)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const int lightBagIndex = 0;  //TODO use actual light bag
    auto* pickedLightBag = &a_LightBags[lightBagIndex * a_NumLightsPerBag];

    for (int i = index; i < a_NumLightBags; i += stride)
    {
        auto* intersectionData = &a_IntersectionData[i];
        auto* reservoir = &a_Reservoirs[i];
        reservoir->Reset();

        //Only sample for intersected pixels.
        if (intersectionData->m_IntersectionT <= 0.f)
        {
            continue;
        }

        auto* ray = &a_RayData[intersectionData->m_RayArrayIndex];
        float3 surfacePosition = ray->m_Origin + ray->m_Direction * intersectionData->m_IntersectionT;
        float3 surfaceNormal;   //TODO get surface normal from the intersection buffer.

        //Generate the amount of samples specified per reservoir.
        for (int sample = 0; sample < a_NumPrimarySamples; ++sample)
        {
            const int pickedLightIndex = 0;//TODO use an actual random index.
            const LightBagEntry pickedEntry = pickedLightBag[pickedLightIndex];
            const TriangleLight light = pickedEntry.light;
            const float initialPdf = pickedEntry.pdf;

            //Generate random UV coordinates. Between 0 and 1.
            const float u = 0.f;    //TODO generate random float between 0 and 1.
            const float v = 0.f;    //TODO generate random float between 0 and 1.

            //Generate a sample with solid angle PDF for this specific pixel.
            LightSample lightSample;
            {
                lightSample.radiance = light.radiance;

                //TODO generate random point according to UV coordinates. This is taking the center for now.
                lightSample.position = (light.p0 + light.p1 + light.p2) / 3.f;
                lightSample.normal = light.normal;

                //Calculate the solid angle for this light on the given surface.
                float3 toLightDir = lightSample.position - surfacePosition;      //Direction from pixel to light.
                float lDistance = length(toLightDir);
                
                //Can't divide by 0.
                if (lDistance <= 0.f)
                {
                    lightSample.solidAnglePdf = 0.f;
                }
                else
                {
                    //Normalize the direction.
                    toLightDir /= lDistance;

                    //Calculate the solid angle PDF.
                    const float dot1 = clamp(dot(toLightDir, surfaceNormal), 0.f, 1.f);           //Lambertian term clamped between 0 and 1. SurfaceN dot ToLight
                    const float dot2 = clamp(dot(-toLightDir, lightSample.normal), 0.f, 1.f);

                    //Geometry term G(x).
                    const float solidAngle = (dot1 * dot2) / (lDistance * lDistance);

                    //TODO take BSDF into account (material diffuse color multiplied with BRDF for in/out direction I think).
                    //BSDF is equal to material color for now.
                    const auto& bsdf = make_float3(1.f);// a_Pixel.material.color;

                    //Light emittance of the light is equal to it's radiance
                    const auto& emittance = light.radiance;

                    //For the light strenghth/bsdf factors, I take the average of each channel to weigh them.
                    const float3 colorFactor = bsdf * emittance;
                    const float averageColor = (colorFactor.x + colorFactor.y + colorFactor.z) / 3.f;

                    lightSample.solidAnglePdf = (averageColor * solidAngle);

                    //Make sure that the solid angle is valid at this point.
                    assert(lightSample.solidAnglePdf >= 0.f);
                }
            }

            //The final PDF for the light in this reservoir is the solid angle divided by the original PDF of the light being chosen based on radiance.
            const auto pdf = lightSample.solidAnglePdf / initialPdf;
            reservoir->Update(lightSample, pdf);
        }

        reservoir->UpdateWeight();
    }
}

__host__ void VisibilityPass(Reservoir* a_Reservoirs, unsigned a_NumReservoirs)
{
    //TODO use this struct
    RestirShadowRay ray;
    ray.index = a_NumReservoirs; //index of the reservoir corresponding to the ray.

    //TODO generate shadow rays and then resolve them in optix. In the optix shader set the reservoir at each rays index to 0 if occluded.
}

__host__ void SpatialNeighbourSampling(Reservoir* a_Reservoirs, Reservoir* a_ReservoirSwapBuffer,
                                       const ReSTIRSettings& a_Settings)
{
    //TODO have a biased and unbiased implementation based on settings.
    //TODO Use the swap buffer as intermediate output. Synchronize between each pass.
}

__host__ void TemporalNeighbourSampling(Reservoir* a_Reservoirs, Reservoir* a_TemporalReservoirs,
                                        const ReSTIRSettings& a_Settings)
{
    //TODO have a biased and unbiased implementation based on settings.
    //TODO Combine the reservoirs by finding temporal sample in buffer based on motion vector. For now use the same index.
}
