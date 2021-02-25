#include "ReSTIRKernels.cuh"

#include "../Shaders/CppCommon/RenderingUtility.h"
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
    PickPrimarySamplesInternal<<<numBlocks, blockSize>>>(a_RayData, a_IntersectionData, a_LightBags, a_Reservoirs, a_Settings);
    cudaDeviceSynchronize();
}

__global__ void PickPrimarySamplesInternal(const WaveFront::RayData* const a_RayData, const WaveFront::IntersectionData* const a_IntersectionData, const LightBagEntry* const a_LightBags, Reservoir* a_Reservoirs, const ReSTIRSettings& a_Settings)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const auto numPixels = a_Settings.width * a_Settings.height;

    //Pick a light bag index based on max indices.
    const int lightBagIndex = 0;  //TODO use actual light bag
    auto* pickedLightBag = &a_LightBags[lightBagIndex * a_Settings.numLightsPerBag];


    //Loop over the pixels
    for (int i = index; i < numPixels; i += stride)
    {
        //Get the intersection data for this pixel.
        auto* intersectionData = &a_IntersectionData[i];

        //If no intersection exists at this pixel, do nothing.
        if(intersectionData->m_IntersectionT <= 0.f)
        {
            continue;
        }

        //The ray that resulted in this intersection.
        auto* ray = &a_RayData[intersectionData->m_RayArrayIndex];

        //Extract the pixel data from the right buffers.
        PixelData pixel;
        pixel.worldPosition = ray->m_Origin + ray->m_Direction * intersectionData->m_IntersectionT;
        pixel.directionIncoming = ray->m_Direction;
        pixel.worldNormal;   //TODO get surface normal from the intersection buffer.
        pixel.diffuse;      //TODO get diffuse color from intersection buffer.
        pixel.roughness;    //TODO get rougghness from intersection.
        pixel.metallic;     //TODO get metallic factor from intersection.

        //For every pixel, update each reservoir.
        for (int reservoirIndex = 0; reservoirIndex < a_Settings.numReservoirsPerPixel; ++reservoirIndex)
        {
            auto* reservoir = &a_Reservoirs[RESERVOIR_INDEX(i, reservoirIndex, a_Settings.numReservoirsPerPixel)];
            reservoir->Reset();

            //Only sample for intersected pixels.
            if (intersectionData->m_IntersectionT <= 0.f)
            {
                continue;
            }

            //Generate the amount of samples specified per reservoir.
            for (int sample = 0; sample < a_Settings.numPrimarySamples; ++sample)
            {
                const int pickedLightIndex = 0;//TODO use an actual random index within thhe light size.
                const LightBagEntry pickedEntry = pickedLightBag[pickedLightIndex];
                const TriangleLight light = pickedEntry.light;
                const float initialPdf = pickedEntry.pdf;

                //Generate random UV coordinates. Between 0 and 1.
                const float u = 0.f;    //TODO generate random float between 0 and 1.
                const float v = 0.f;    //TODO generate random float between 0 and 1.

                //Generate a sample with solid angle PDF for this specific pixel.
                LightSample lightSample;
                {
                    //Fill the light with the right settings.
                    lightSample.radiance = light.radiance;
                    lightSample.normal = light.normal;
                    lightSample.area = light.area;
                    //TODO generate random point according to UV coordinates. This is taking the center for now.
                    lightSample.position = (light.p0 + light.p1 + light.p2) / 3.f;

                    //Calculate the PDF for this pixel and light.
                    Resample(&lightSample, &pixel, &lightSample);
                }

                //The final PDF for the light in this reservoir is the solid angle divided by the original PDF of the light being chosen based on radiance.
                const auto pdf = lightSample.solidAnglePdf / initialPdf;
                reservoir->Update(lightSample, pdf);
            }

            reservoir->UpdateWeight();
        }
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

__device__ void CombineUnbiased(int a_PixelIndex, int a_Count, int a_MaxReservoirDepth, Reservoir* a_Reservoirs, PixelData* a_ToCombine)
{

    for (int depth = 0; depth < a_MaxReservoirDepth; ++depth)
    {
        Reservoir output;
        int sampleCountSum = 0;

        for (int index = 0; index < a_Count; ++index)
        {
            auto* otherReservoir = &a_Reservoirs[RESERVOIR_INDEX(index, depth, a_MaxReservoirDepth)];
            LightSample resampled;
            Resample(&otherReservoir->sample, &a_ToCombine[index], &resampled);

            const float weight = static_cast<float>(otherReservoir->sampleCount) * otherReservoir->weight * resampled.solidAnglePdf;

            output.Update(resampled, weight);

            sampleCountSum += otherReservoir->sampleCount;
        }

        output.sampleCount = sampleCountSum;

        //Weigh against other pixels to remove bias from their solid angle by re-sampling.
        int correction = 0;

        for (int index = 0; index < a_Count; ++index)
        {
            auto* otherPixel = &a_ToCombine[index];
            LightSample resampled;
            Resample(&output.sample, otherPixel, &resampled);

            if (resampled.solidAnglePdf > 0)
            {
                correction += a_Reservoirs[RESERVOIR_INDEX(otherPixel->index, depth, a_MaxReservoirDepth)].sampleCount;
            }
        }

        //TODO Shadow ray is shot here in ReSTIR to check visibility at every resampled pixel.


        //TODO I don't understand this part fully, but it's in the pseudocode of ReSTIR. Dive into it when I have time.
        const float m = 1.f / fmaxf(static_cast<float>(correction), MINFLOAT);
        output.weight = (1.f / fmaxf(output.sample.solidAnglePdf, MINFLOAT)) * (m * output.weightSum);

        //Store the output reservoir for the pixel.
        a_Reservoirs[RESERVOIR_INDEX(a_PixelIndex, depth, a_MaxReservoirDepth)];
    }
}

__device__ void CombineBiased(int a_PixelIndex, int a_Count, int a_MaxReservoirDepth, Reservoir* a_Reservoirs, PixelData* a_ToCombine)
{
    //Loop over every depth.
    for (int depth = 0; depth < a_MaxReservoirDepth; ++depth)
    {
        Reservoir output;
        int sampleCountSum = 0;

        //Iterate over the intersection data to combine.
        for (int i = 0; i < a_Count; ++i)
        {
            auto* pixel = &a_ToCombine[i];
            auto* reservoir = &a_Reservoirs[RESERVOIR_INDEX(pixel->index, depth, a_MaxReservoirDepth)];

            LightSample resampled;
            Resample(&reservoir->sample, pixel, &resampled);

            const float weight = static_cast<float>(reservoir->sampleCount) * reservoir->weight * resampled.solidAnglePdf;

            assert(resampled.solidAnglePdf >= 0.f);

            output.Update(resampled, weight);

            sampleCountSum += reservoir->sampleCount;
        }

        //Update the sample 
        output.sampleCount = sampleCountSum;
        output.UpdateWeight();

        assert(output.weight >= 0.f && output.weightSum >= 0.f);

        //Override the reservoir for the output at this depth.
        a_Reservoirs[RESERVOIR_INDEX(a_PixelIndex, depth, a_MaxReservoirDepth)] = output;
    }
}

__device__ void Resample(LightSample* a_Input, const PixelData* a_PixelData, LightSample* a_Output)
{
    *a_Output = *a_Input;

    float3 pixelToLightDir = a_Input->position - a_PixelData->worldPosition;                                   //Direction from pixel to light.
    const float lDistance = length(pixelToLightDir);                                                               //Light distance from pixel.
    pixelToLightDir /= lDistance;                                                                                          //Normalize.
    const float cosIn = clamp(dot(pixelToLightDir, a_PixelData->worldNormal), 0.f, 1.f);           //Lambertian term clamped between 0 and 1. SurfaceN dot ToLight
    const float cosOut = clamp(dot(a_Input->normal, -pixelToLightDir), 0.f, 1.f);    //Light normal at sample point dotted with light direction. Invert light dir for this (light to pixel instead of pixel to light)

    //Light is not facing towards the surface or too close to the surface.
    if(cosIn <= 0 || cosOut <= 0 || lDistance <= 0.01f)
    {
        a_Output->solidAnglePdf = 0;
        return;
    }

    //Geometry term G(x).
    const float solidAngle = (cosOut * a_Input->area) / (lDistance * lDistance);

    //BSDF is equal to material color for now.
    const auto brdf = MicrofacetBRDF(-pixelToLightDir, a_PixelData->directionIncoming, a_PixelData->worldNormal, a_PixelData->diffuse, a_PixelData->metallic, a_PixelData->roughness);

    //The unshadowed contribution (contributed if no obstruction is between the light and surface) takes the BRDF,
    //geometry factor and solid angle into account. Also the light radiance.
    //The only thing missing from this is the scaling with the rest of the scene based on the reservoir PDF.
    const auto unshadowedPathContribution = brdf * solidAngle * cosIn * a_Output->radiance;
    a_Output->unshadowedPathContribution = unshadowedPathContribution;

    //For the PDF, I take the unshadowed path contribution as a single float value. Average for now.
    //TODO might be better to instead take the max value? Ask Jacco.
    a_Output->solidAnglePdf = (unshadowedPathContribution.x + unshadowedPathContribution.y + unshadowedPathContribution.z) / 3.f;
}
