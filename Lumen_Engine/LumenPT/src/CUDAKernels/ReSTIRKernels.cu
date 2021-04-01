#include "ReSTIRKernels.cuh"

#include "../Shaders/CppCommon/RenderingUtility.h"
#include "../Shaders/CppCommon/WaveFrontDataStructs.h"
#include <cuda_runtime_api.h>
#include <cuda/device_atomic_functions.h>

#include "../Framework/CudaUtilities.h"

#define CUDA_BLOCK_SIZE 512

__host__ void ResetReservoirs(int a_NumReservoirs, Reservoir* a_ReservoirPointer)
{
    //Call in parallel.
    const int blockSize = CUDA_BLOCK_SIZE;
    const int numBlocks = (a_NumReservoirs + blockSize - 1) / blockSize;
    ResetReservoirInternal<<<numBlocks, blockSize>>>(a_NumReservoirs, a_ReservoirPointer);

    //TODO: Wait after every task may not be needed.Check if it is required between kernel calls.
    cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;
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

__host__ void FillCDF(CDF* a_Cdf, const WaveFront::TriangleLight* a_Lights, unsigned a_LightCount)
{
    //TODO: This is not efficient single threaded.
    //TODO: Use this: https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda

    //First reset the CDF on the GPU.
    ResetCDF<<<1,1>>>(a_Cdf);
    cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;

    //Run from one thread because it's not thread safe to append the sum of each element.
    FillCDFInternal <<<1, 1>>> (a_Cdf, a_Lights, a_LightCount);
    cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;
}

__global__ void ResetCDF(CDF* a_Cdf)
{
    a_Cdf->Reset();
}

__global__ void FillCDFInternal(CDF* a_Cdf, const WaveFront::TriangleLight* a_Lights, unsigned a_LightCount)
{
    for (int i = 0; i < a_LightCount; ++i)
    {
        //Weight is the average illumination for now. Could take camera into account.
        const float3 radiance = a_Lights[i].radiance;
        a_Cdf->Insert((radiance.x + radiance.y + radiance.z) / 3.f);
    }
}

__host__ void FillLightBags(unsigned a_NumLightBags, unsigned a_NumLightsPerBag, CDF* a_Cdf, LightBagEntry* a_LightBagPtr, const WaveFront::TriangleLight* a_Lights, const std::uint32_t a_Seed)
{
    const unsigned numLightsTotal = a_NumLightBags * a_NumLightsPerBag;
    const int blockSize = CUDA_BLOCK_SIZE;
    const int numBlocks = (numLightsTotal + blockSize - 1) / blockSize;
    FillLightBagsInternal <<<numBlocks, blockSize >>>(a_NumLightBags, a_NumLightsPerBag, a_Cdf, a_LightBagPtr, a_Lights, a_Seed);
    cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;
}

__global__ void FillLightBagsInternal(unsigned a_NumLightBags, unsigned a_NumLightsPerBag, CDF* a_Cdf, LightBagEntry* a_LightBagPtr, const WaveFront::TriangleLight* a_Lights, const std::uint32_t a_Seed)
{
    const unsigned numLightsTotal = a_NumLightBags * a_NumLightsPerBag;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < numLightsTotal; i += stride)
    {
        //Generate a random float between 0 and 1.
        auto seed = WangHash(a_Seed + WangHash(i));
        const float random = RandomFloat(seed);

        //Store the pdf and light in the light bag.
        unsigned lIndex;
        float pdf;
        a_Cdf->Get(random, lIndex, pdf);
        a_LightBagPtr[i] = LightBagEntry{a_Lights[lIndex], pdf};
    }
}

__host__ void PickPrimarySamples(const LightBagEntry* const a_LightBags, Reservoir* a_Reservoirs, const ReSTIRSettings& a_Settings, const WaveFront::SurfaceData * const a_PixelData, const std::uint32_t a_Seed)
{
    /*
     * This functions uses a single light bag per block.
     * This means that all threads within a block operate on the same light bag data in the cache.
     * The pixel access is also optimized for cache hits per block.
     */
    const auto numReservoirs = (a_Settings.width * a_Settings.height * a_Settings.numReservoirsPerPixel);
    const int blockSize = CUDA_BLOCK_SIZE;
    const int numBlocks = (numReservoirs + blockSize - 1) / blockSize;
    PickPrimarySamplesInternal<<<numBlocks, blockSize>>>
    (
        a_LightBags,
        a_Reservoirs, a_Settings.numPrimarySamples,
        numReservoirs,
        a_Settings.numReservoirsPerPixel,
        a_Settings.numLightBags,
        a_Settings.numLightsPerBag,
        a_PixelData,
        a_Seed);
    cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;
}

__global__ void PickPrimarySamplesInternal(const LightBagEntry* const a_LightBags, Reservoir* a_Reservoirs, unsigned a_NumPrimarySamples, unsigned a_NumReservoirs, unsigned a_NumReservoirsPerPixel, unsigned a_NumLightBags, unsigned a_NumLightsPerBag, const WaveFront::SurfaceData * const a_PixelData, const std::uint32_t a_Seed)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= a_NumReservoirs)
    {
        return;
    }

    //Seed for this thread index.
    //Seed is the same for each block so that they all get the same light bag.

    //This would pick the light bag index based on the block index. Multiple blocks may share a cache though.
    //auto lBagSeed = WangHash(a_Seed + blockIdx.x);

    //This picks the light bag index based on the SM id. Every SM has its own L1 cache. Multiple blocks may execute on one SM at a time.
    //This line thus causes the L1 cache to only access a single light bag, which stops a lot of cache misses.
    //TODO: This means that depending on the blocks per SM, a single light bag may be used for multiple rows of pixels. Ensure no artifacts occur (difference in light in interleaved rows).
    //TODO: If this occurs, consider either changing back (performance loss) or interleaving the pixels in a way where a row of 256 pixels is actually a 16x16 area in the screen.
    auto lBagSeed = WangHash(a_Seed + __mysmid());

    const float random = RandomFloat(lBagSeed);

    //Generate between 0 and 1, then round and pick a light bag index based on the total light bag amount.
    const int lightBagIndex = static_cast<int>(roundf(static_cast<float>(a_NumLightBags - 1) * random));
    auto* pickedLightBag = &a_LightBags[lightBagIndex * a_NumLightsPerBag];

    const auto pixelIndex = index / a_NumReservoirsPerPixel;

    //The current pixel index.
    const WaveFront::SurfaceData* pixel = &a_PixelData[pixelIndex];

    //If no intersection exists at this pixel, do nothing. Emissive surfaces are also excluded.
    if(pixel->m_IntersectionT <= 0.f || pixel->m_Emissive)
    {
        return;
    }

    //Base Seed depending on pixel index.
    auto seed = WangHash(a_Seed + WangHash(index));

    //Fresh reservoir to start with.
    Reservoir fresh;

    int pickedLightIndex;
    LightBagEntry pickedEntry;
    WaveFront::TriangleLight light;
    float initialPdf;

    //Generate the amount of samples specified per reservoir.
    for (int sample = 0; sample < a_NumPrimarySamples; ++sample)
    {
        //Random number using the pixel id.
        const float r = RandomFloat(seed);

        pickedLightIndex = static_cast<int>(roundf(static_cast<float>(a_NumLightsPerBag - 1) * r));
        pickedEntry = pickedLightBag[pickedLightIndex];
        light = pickedEntry.light;
        initialPdf = pickedEntry.pdf;

        //Generate random UV coordinates. Between 0 and 1.
        const float u = RandomFloat(seed);  //Seed is altered after each shift, which makes it work with the same uint.
        const float v = RandomFloat(seed) * (1.f - u);

        //Generate a sample with solid angle PDF for this specific pixel.
        LightSample lightSample;
        {
            //Fill the light with the right settings.
            lightSample.radiance = light.radiance;
            lightSample.normal = light.normal;
            lightSample.area = light.area;

            //Generate the position on the triangle uniformly. TODO: this may not be uniform?
            float3 arm1 = light.p1 - light.p0;
            float3 arm2 = light.p2 - light.p0;
            lightSample.position = light.p0 + (arm1 * u) + (arm2 * v);

            //Calculate the PDF for this pixel and light.
            Resample(&lightSample, pixel, &lightSample);
        }

        //The final PDF for the light in this reservoir is the solid angle divided by the original PDF of the light being chosen based on radiance.
        //Dividing scales the value up.
        const auto pdf = lightSample.solidAnglePdf / initialPdf;
        fresh.Update(lightSample, pdf, seed);
    }

    //Finally update the reservoir weight.
    fresh.UpdateWeight();

    //Override the old reservoir.
    auto* reservoir = &a_Reservoirs[index];
    *reservoir = fresh;
}

__host__ unsigned int GenerateReSTIRShadowRays(MemoryBuffer* a_AtomicBuffer, Reservoir* a_Reservoirs, const WaveFront::SurfaceData* a_PixelData, unsigned a_NumPixels)
{
    //Counter that is atomically incremented. Copy it to the GPU.
    WaveFront::ResetAtomicBuffer<RestirShadowRay>(a_AtomicBuffer);

    auto devicePtr = a_AtomicBuffer->GetDevicePtr<WaveFront::AtomicBuffer<RestirShadowRay>>();

    //Call in parallel.
    const int blockSize = CUDA_BLOCK_SIZE;
    const int numBlocks = (a_NumPixels + blockSize - 1) / blockSize;
    GenerateShadowRay<<<numBlocks, blockSize>>> (devicePtr, a_Reservoirs, a_PixelData, a_NumPixels);
    cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;

    //Copy value back to the CPU.
    return WaveFront::GetAtomicCounter<RestirShadowRay>(a_AtomicBuffer);
}

__global__ void GenerateShadowRay(WaveFront::AtomicBuffer<RestirShadowRay>* a_AtomicBuffer, Reservoir* a_Reservoirs, const WaveFront::SurfaceData* a_PixelData, unsigned a_NumPixels)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int pixel = index; pixel < a_NumPixels; pixel += stride)
    {
        const WaveFront::SurfaceData* pixelData = &a_PixelData[pixel];

        //Only run for valid intersections.
        if(pixelData->m_IntersectionT > 0.f && !pixelData->m_Emissive)
        {
            float3 pixelPosition = pixelData->m_Position;

            //Run for every reservoir for the pixel.
            for(int depth = 0; depth < ReSTIRSettings::numReservoirsPerPixel; ++depth)
            {
                //If the reservoir has a weight, add a shadow ray.
                Reservoir* reservoir = &a_Reservoirs[RESERVOIR_INDEX(pixel, depth, ReSTIRSettings::numReservoirsPerPixel)];

                if(reservoir->weight > 0.f)
                {
                    float3 pixelToLight = (reservoir->sample.position - pixelPosition);
                    const float l = length(pixelToLight);
                    pixelToLight /= l;

                    assert(fabsf(length(pixelToLight) - 1.f) <= FLT_EPSILON * 5.f);
                    //printf("Length: %f\n", length(pixelToLight));

                    RestirShadowRay ray;
                    ray.index = pixel;
                    ray.direction = pixelToLight;
                    ray.origin = pixelPosition;
                    ray.distance = l - 0.05f; //Make length a little bit shorter to prevent self-shadowing.

                    a_AtomicBuffer->Add(&ray);
                }
            }            
        }
    }
}

__host__ void SpatialNeighbourSampling(Reservoir* a_Reservoirs, Reservoir* a_SwapBuffer, const WaveFront::SurfaceData* a_PixelData, const std::uint32_t a_Seed, uint2 a_Dimensions)
{
    const unsigned numPixels = a_Dimensions.x * a_Dimensions.y;
    const int blockSize = CUDA_BLOCK_SIZE;
    const int numBlocks = (numPixels + blockSize - 1) / blockSize;

    Reservoir* fromBuffer = a_Reservoirs;
    Reservoir* toBuffer = a_SwapBuffer;

    //Synchronize between each swap, and then swap the buffers.
    for (int iteration = 0; iteration < ReSTIRSettings::numSpatialIterations; ++iteration)
    {

        SpatialNeighbourSamplingInternal<<<numBlocks, blockSize >>>(fromBuffer, toBuffer, a_PixelData, a_Seed, a_Dimensions);
        cudaDeviceSynchronize();
        CHECKLASTCUDAERROR;

        //Swap the pointers for in and output.
        Reservoir* temp = fromBuffer;
        fromBuffer = toBuffer;
        toBuffer = temp;
    }

}

//TODO: The bug is that it's swapping buffers per thread in the middle all over the place. Should finish first wave, then swap and continue.
__global__ void SpatialNeighbourSamplingInternal(Reservoir* a_Reservoirs, Reservoir* a_SwapBuffer,
    const WaveFront::SurfaceData* a_PixelData, const std::uint32_t a_Seed, uint2 a_Dimensions)
{
    const unsigned numPixels = a_Dimensions.x * a_Dimensions.y;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    //Storage for reservoirs and pixels to be combined.
    const WaveFront::SurfaceData* toCombinePixelData[ReSTIRSettings::numSpatialSamples + 1];
    Reservoir* toCombineReservoirs[ReSTIRSettings::numSpatialSamples + 1];

    //Loop over the pixels.
    for (int i = index; i < numPixels; i += stride)
    {
        //The seed unique to this pixel.
        auto seed =  WangHash(a_Seed + i);

        toCombinePixelData[0] = &a_PixelData[i];

        //Only run when there's an intersection for this pixel.
        if (toCombinePixelData[0]->m_IntersectionT > 0.f && !toCombinePixelData[0]->m_Emissive)
        {
            //TODO maybe store this information inside the pixel. Could calculate it once at the start of the frame.
            const int y = i / a_Dimensions.x;
            const int x = i - (y * a_Dimensions.x);

            for (int depth = 0; depth < ReSTIRSettings::numReservoirsPerPixel; ++depth)
            {
                toCombineReservoirs[0] = &a_Reservoirs[RESERVOIR_INDEX(i, depth, ReSTIRSettings::numReservoirsPerPixel)];

                int count = 1;

                //TODO move this loop up maybe? Use same neighbour for each depth since values are different anyways. I don't think that changes the
                //TODO average of the calculation.
                for (int neighbour = 1; neighbour <= ReSTIRSettings::numSpatialSamples; ++neighbour)
                {
                    //TODO This generates a square rn. Make it within a circle.
                    const int neighbourY = round((RandomFloat(seed) * 2.f - 1.f) * static_cast<float>(ReSTIRSettings::spatialSampleRadius)) + y;
                    const int neighbourX = round((RandomFloat(seed) * 2.f - 1.f) * static_cast<float>(ReSTIRSettings::spatialSampleRadius)) + x;

                    //Only run if within image bounds.
                    if(neighbourX >= 0 && neighbourX < a_Dimensions.x && neighbourY >= 0 && neighbourY < a_Dimensions.y)
                    {
                        //Index of the neighbouring pixel in the pixel array.
                        const int neighbourIndex = PIXEL_INDEX(neighbourX, neighbourY, a_Dimensions.x);

                        //Ensure no out of bounds neighbours are selected.
                        assert(neighbourIndex < numPixels && neighbourIndex >= 0); 

                        Reservoir* pickedReservoir = &a_Reservoirs[RESERVOIR_INDEX(neighbourIndex, depth, ReSTIRSettings::numReservoirsPerPixel)];
                        const WaveFront::SurfaceData* pickedPixel = &a_PixelData[neighbourIndex];

                        //Only run for valid depths and non-emissive surfaces.
                        if(pickedPixel->m_IntersectionT > 0.f && !pickedPixel->m_Emissive)
                        {
                            //Gotta stay positive.
                            assert(pickedReservoir->weight >= 0.f);

                            //Discard samples that are too different.
                            const float depth1 = pickedPixel->m_IntersectionT;
                            const float depth2 = toCombinePixelData[0]->m_IntersectionT;
                            const float depthDifPct = fabs(depth1 - depth2) / ((depth1 + depth2) / 2.f);

                            const float angleDif = dot(pickedPixel->m_Normal, toCombinePixelData[0]->m_Normal);	//Between 0 and 1 (0 to 90 degrees). 
                            static constexpr float MAX_ANGLE_COS = 0.72222222223f;	//Dot product is cos of the angle. If higher than this value, it's within 25 degrees.

                            if (depthDifPct < 0.10f && angleDif > MAX_ANGLE_COS)
                            {
                                toCombineReservoirs[count] = pickedReservoir;
                                toCombinePixelData[count] = pickedPixel;
                                ++count;
                            }
                        }
                    }
                }

                //The output location.
                Reservoir* output = &a_SwapBuffer[RESERVOIR_INDEX(i, depth, ReSTIRSettings::numReservoirsPerPixel)];

                //If valid reservoirs to combine were found, combine them.
                if (count > 1)
                {
                    if(ReSTIRSettings::enableBiased)
                    {
                        CombineBiased(output, count, toCombineReservoirs, toCombinePixelData[0], seed);
                    }
                    else
                    {
                        CombineUnbiased(output, toCombinePixelData[0], count, toCombineReservoirs, toCombinePixelData, seed);
                    }
                }
                //Not enough reservoirs to combine, but still data needs to be passed on.
                else
                {
                    //Copy the reservoir over directly without any combining.
                    *output = *toCombineReservoirs[0];
                }
            }
        }
    }
}


__host__ void TemporalNeighbourSampling(
    Reservoir* a_CurrentReservoirs,
    Reservoir* a_PreviousReservoirs,
    const WaveFront::SurfaceData* a_CurrentPixelData,
    const WaveFront::SurfaceData* a_PreviousPixelData,
    const std::uint32_t a_Seed,
    unsigned a_NumPixels,
    uint2 a_Dimensions,
    WaveFront::MotionVectorBuffer* a_MotionVectorBuffer
)
{
    const int blockSize = CUDA_BLOCK_SIZE;
    const int numBlocks = (a_NumPixels + blockSize - 1) / blockSize;

    CombineTemporalSamplesInternal << <numBlocks, blockSize >> > (a_CurrentReservoirs, a_PreviousReservoirs,
                                                                  a_CurrentPixelData, a_PreviousPixelData, a_Seed, a_NumPixels, a_Dimensions, a_MotionVectorBuffer);
    cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;
}


__global__ void CombineTemporalSamplesInternal(
    Reservoir* a_CurrentReservoirs,
    Reservoir* a_PreviousReservoirs,
    const WaveFront::SurfaceData* a_CurrentPixelData,
    const WaveFront::SurfaceData* a_PreviousPixelData,
    const std::uint32_t a_Seed,
    unsigned a_NumPixels,
    uint2 a_Dimensions,
    WaveFront::MotionVectorBuffer* a_MotionVectorBuffer
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    Reservoir* toCombine[2];
    const WaveFront::SurfaceData* pixelPointers[2];

    for (int i = index; i < a_NumPixels; i += stride)
    {
        const auto velocity = -a_MotionVectorBuffer->GetMotionVectorData(i).m_Velocity;
        const int movedX = roundf((float)a_Dimensions.x * velocity.x);
        const int movedY = roundf((float)a_Dimensions.y * velocity.y);

        int y = i / a_Dimensions.x;
        int x = i - (y * a_Dimensions.x);
        y += movedY;
        x += movedX;

        //Ensure 
        int temporalIndex = i;
        if(y >= 0 && y < a_Dimensions.y && x >= 0 && x < a_Dimensions.x)
        {
            temporalIndex = PIXEL_INDEX(x, y, a_Dimensions.x);
        }

        const WaveFront::SurfaceData* surface = &a_CurrentPixelData[i];

        pixelPointers[0] = &a_PreviousPixelData[temporalIndex];
        pixelPointers[1] = surface;

        //Ensure that the depth of both samples is valid, and then combine them at each depth.
        if (pixelPointers[0]->m_IntersectionT > 0.f && pixelPointers[1]->m_IntersectionT > 0.f && !pixelPointers[0]->m_Emissive && !pixelPointers[1]->m_Emissive)
        {
            //For every reservoir at the current pixel.
            for (int depth = 0; depth < ReSTIRSettings::numReservoirsPerPixel; ++depth)
            {
                toCombine[0] = &a_PreviousReservoirs[RESERVOIR_INDEX(temporalIndex, depth, ReSTIRSettings::numReservoirsPerPixel)];
                toCombine[1] = &a_CurrentReservoirs[RESERVOIR_INDEX(i, depth, ReSTIRSettings::numReservoirsPerPixel)];

                //Discard samples that are too different.
                float depth1 = pixelPointers[0]->m_IntersectionT;
                float depth2 = pixelPointers[1]->m_IntersectionT;
                float depthDifPct = fabs(depth1 - depth2) / ((depth1 + depth2) / 2.f);

                const float angleDif = dot(pixelPointers[0]->m_Normal, pixelPointers[1]->m_Normal);	//Between 0 and 1 (0 to 90 degrees). 
                static constexpr float MAX_ANGLE_COS = 0.72222222223f;	//Dot product is cos of the angle. If higher than this value, it's within 25 degrees.

                //Only do something if the samples are not vastly different.
                if (depthDifPct < 0.10f && angleDif > MAX_ANGLE_COS)
                {
                    //Cap sample count at 20x current to reduce temporal influence. Would grow infinitely large otherwise.
                    toCombine[0]->sampleCount = min(toCombine[0]->sampleCount, toCombine[1]->sampleCount * 20);

                    Reservoir* output = &a_CurrentReservoirs[RESERVOIR_INDEX(i, depth, ReSTIRSettings::numReservoirsPerPixel)];

                    if (ReSTIRSettings::enableBiased)
                    {
                        CombineBiased(output, 2, toCombine, surface, WangHash(a_Seed + i));
                    }
                    else
                    {
                        CombineUnbiased(output, surface, 2, toCombine, pixelPointers, WangHash(a_Seed + i));
                    }
                }
            }
        }
    }
}

__device__ __inline__ void CombineUnbiased(Reservoir* a_OutputReservoir, const WaveFront::SurfaceData* a_OutputSurfaceData, int a_Count, Reservoir** a_Reservoirs,
    const WaveFront::SurfaceData** a_ToCombine, const std::uint32_t a_Seed)
{
    //Ensure enough reservoirs are passed.
    assert(a_Count > 1);

    Reservoir output;
    int sampleCountSum = 0;

    for (int index = 0; index < a_Count; ++index)
    {
        //TODO: possible optimization here: no resampling required for the pixel being merged into if applicable.
        auto* otherReservoir = a_Reservoirs[index];
        LightSample resampled;
        Resample(&otherReservoir->sample, a_OutputSurfaceData, &resampled);

        assert(otherReservoir->weight >= 0.f);
        assert(otherReservoir->sampleCount >= 0);
        assert(otherReservoir->weightSum >= 0.f);
        assert(otherReservoir->sample.solidAnglePdf >= 0.f);

        const float weight = static_cast<float>(otherReservoir->sampleCount) * otherReservoir->weight * resampled.solidAnglePdf;

        output.Update(resampled, weight, a_Seed);

        sampleCountSum += otherReservoir->sampleCount;
    }

    output.sampleCount = sampleCountSum;

    //Weigh against other pixels to remove bias from their solid angle by re-sampling.
    int correction = 0;

    for (int index = 0; index < a_Count; ++index)
    {
        auto* otherPixel = a_ToCombine[index];
        LightSample resampled;
        Resample(&output.sample, otherPixel, &resampled);

        if (resampled.solidAnglePdf > 0)
        {
            correction += a_Reservoirs[index]->sampleCount;
        }
    }


    //TODO Shadow ray is shot here in ReSTIR to check visibility at every resampled pixel.


    //TODO I don't understand this part fully, but it's in the pseudocode of ReSTIR. Dive into it when I have time.
    const float m = 1.f / fmaxf(static_cast<float>(correction), MINFLOAT);
    output.weight = (1.f / fmaxf(output.sample.solidAnglePdf, MINFLOAT)) * (m * output.weightSum);

    assert(output.weight >= 0.f);
    assert(output.weightSum >= 0.f);
    assert(output.sampleCount >= 0);
    assert(!isnan(output.weight));
    assert(!isnan(output.weightSum));
    assert(!isnan(output.sample.solidAnglePdf));
    assert(output.sample.solidAnglePdf >= 0.f);
    assert(!isinf(output.sample.solidAnglePdf));
    assert(!isinf(output.weight));
    assert(!isinf(output.weightSum));

    //Store the output reservoir for the pixel.
    *a_OutputReservoir = output;
}

__device__ __inline__ void CombineBiased(Reservoir* a_OutputReservoir, int a_Count, Reservoir** a_Reservoirs,
    const WaveFront::SurfaceData* a_SurfaceData, const std::uint32_t a_Seed)
{
    //Ensure enough reservoirs are passed.
    assert(a_Count > 1);

    //Loop over every depth.
    Reservoir output;
    long long int sampleCountSum = 0;

    //Iterate over the intersection data to combine.
    for (int i = 0; i < a_Count; ++i)
    {
        auto* reservoir = a_Reservoirs[i];

        assert(reservoir->weight >= 0.f);
        assert(reservoir->sampleCount >= 0);
        assert(reservoir->weightSum >= 0.f);
        assert(reservoir->sample.solidAnglePdf >= 0.f);
        
        LightSample resampled;
        Resample(&(reservoir->sample), a_SurfaceData, &resampled);

        const float weight = static_cast<float>(reservoir->sampleCount) * reservoir->weight * resampled.solidAnglePdf;

        assert(resampled.solidAnglePdf >= 0.f);

        output.Update(resampled, weight, a_Seed);

        sampleCountSum += reservoir->sampleCount;
    }

    //Update the sample 
    output.sampleCount = sampleCountSum;
    output.UpdateWeight();

    assert(output.weight >= 0.f);
    assert(output.weightSum >= 0.f);
    assert(output.sampleCount >= 0);
    assert(!isnan(output.weight));
    assert(!isnan(output.weightSum));
    assert(!isnan(output.sample.solidAnglePdf));
    assert(output.sample.solidAnglePdf >= 0.f);
    assert(!isinf(output.sample.solidAnglePdf));
    assert(!isinf(output.weight));
    assert(!isinf(output.weightSum));

    //Override the reservoir for the output at this depth.
    *a_OutputReservoir = output;
}

__device__ __inline__ void Resample(LightSample* a_Input, const WaveFront::SurfaceData* a_PixelData, LightSample* a_Output)
{
    *a_Output = *a_Input;

    float3 pixelToLightDir = a_Input->position - a_PixelData->m_Position;
    //Direction from pixel to light.
    const float lDistance = length(pixelToLightDir);
    //Light distance from pixel.
    pixelToLightDir /= lDistance;
    //Normalize.
    const float cosIn = fmax(dot(pixelToLightDir, a_PixelData->m_Normal), 0.f);
    //Lambertian term clamped between 0 and 1. SurfaceN dot ToLight
    const float cosOut = fmax(dot(a_Input->normal, -pixelToLightDir), 0.f);
    //Light normal at sample point dotted with light direction. Invert light dir for this (light to pixel instead of pixel to light)

    //Light is not facing towards the surface or too close to the surface.
    if(cosIn <= 0 || cosOut <= 0 || lDistance <= 0.01f)
    {
        a_Output->solidAnglePdf = 0;
        return;
    }

    //Geometry term G(x).
    const float solidAngle = (cosOut * a_Input->area) / (lDistance * lDistance);

    //BSDF is equal to material color for now.
    const auto brdf = MicrofacetBRDF(pixelToLightDir, -a_PixelData->m_IncomingRayDirection, a_PixelData->m_Normal,
                                     a_PixelData->m_Color, a_PixelData->m_Metallic, a_PixelData->m_Roughness);

    //The unshadowed contribution (contributed if no obstruction is between the light and surface) takes the BRDF,
    //geometry factor and solid angle into account. Also the light radiance.
    //The only thing missing from this is the scaling with the rest of the scene based on the reservoir PDF.
    //Note: No need to multiply with transport factor because this is depth 0. It is always {1, 1, 1}.
    const auto unshadowedPathContribution = brdf * solidAngle * cosIn * a_Output->radiance;
    a_Output->unshadowedPathContribution = unshadowedPathContribution;

    assert(unshadowedPathContribution.x >= 0 && unshadowedPathContribution.y >= 0 && unshadowedPathContribution.z >= 0);

    //For the PDF, I take the unshadowed path contribution as a single float value. Average for now.
    //TODO: Maybe use the human eye for scaling (green weighed more).
    a_Output->solidAnglePdf = (unshadowedPathContribution.x + unshadowedPathContribution.y + unshadowedPathContribution.
        z) / 3.f;
}

__host__ void GenerateWaveFrontShadowRays(Reservoir* a_Reservoirs, const WaveFront::SurfaceData* a_PixelData, WaveFront::AtomicBuffer<WaveFront::ShadowRayData>* a_AtomicBuffer, unsigned a_NumPixels)
{
    //Call in parallel.
    const int blockSize = CUDA_BLOCK_SIZE;
    const int numBlocks = (a_NumPixels + blockSize - 1) / blockSize;

    //Separate invocations for each depth to add a stride.
    for (int depth = 0; depth < ReSTIRSettings::numReservoirsPerPixel; ++depth)
    {
        GenerateWaveFrontShadowRaysInternal << <numBlocks, blockSize >> > (a_Reservoirs, a_PixelData, a_AtomicBuffer, a_NumPixels, depth);
    }
    cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;
}

__global__ void GenerateWaveFrontShadowRaysInternal(Reservoir* a_Reservoirs, const WaveFront::SurfaceData* a_PixelData, WaveFront::AtomicBuffer<WaveFront::ShadowRayData>* a_AtomicBuffer, unsigned a_NumPixels, unsigned a_Depth)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < a_NumPixels; i += stride)
    {
        const WaveFront::SurfaceData* pixel = &a_PixelData[i];

        //Only generate shadow rays for pixels that hit a surface that is not emissive.
        if(pixel->m_IntersectionT > 0.f && !pixel->m_Emissive)
        {
            /*
             * TODO
             * Note: This currently divides the expected contribution per reservoir by the amount of reservoirs.
             * It's essentially like scaling down so that the total adds up to 100% if all shadow rays pass.
             * This does shoot one shadow ray per reservoir, but I think that's needed for accurate results.
             * If we are really desperate we could average the reservoir results and then send a single shadow ray.
             */

            //Get the contribution and scale it down based on the number of reservoirs.
            Reservoir* reservoir = &a_Reservoirs[RESERVOIR_INDEX(i, a_Depth, ReSTIRSettings::numReservoirsPerPixel)];

            //Only send shadow rays for reservoirs that have a valid sample.
            if(reservoir->weight > 0)
            {
                float3 contribution = (reservoir->sample.unshadowedPathContribution * (reservoir->weight / static_cast<float>(ReSTIRSettings::numReservoirsPerPixel)));

                //Generate a ray for this particular reservoir.
                float3 toLightDir = reservoir->sample.position - pixel->m_Position;
                const float l = length(toLightDir);
                toLightDir /= l;

                //TODO: add stride between these.

                //TODO ensure no shadow acne.
                //TODO: Pass pixel index to shadow ray data.
                auto data = WaveFront::ShadowRayData{ pixel->m_Index, pixel->m_Position, toLightDir, l - 0.005f, contribution, WaveFront::LightChannel::DIRECT };
                a_AtomicBuffer->Add(&data);
            }
        }
    }
}

uint32_t __mysmid()
{
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

