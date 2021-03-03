#include "ReSTIRKernels.cuh"

#include "../Shaders/CppCommon/RenderingUtility.h"
#include "../Shaders/CppCommon/WaveFrontDataStructs.h"
#include <cuda_runtime_api.h>
#include <cuda/device_atomic_functions.h>

#define CUDA_BLOCK_SIZE 256

__host__ void ResetReservoirs(int a_NumReservoirs, Reservoir* a_ReservoirPointer)
{
    //Call in parallel.
    const int blockSize = CUDA_BLOCK_SIZE;
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

__host__ void FillLightBags(unsigned a_NumLightBags, CDF* a_Cdf, LightBagEntry* a_LightBagPtr, TriangleLight* a_Lights, const std::uint32_t a_Seed)
{
    const int blockSize = CUDA_BLOCK_SIZE;
    const int numBlocks = (a_NumLightBags + blockSize - 1) / blockSize;
    FillLightBagsInternal <<<numBlocks, blockSize >>>(a_NumLightBags, a_Cdf, a_LightBagPtr, a_Lights, a_Seed);
    cudaDeviceSynchronize();
}

__global__ void FillLightBagsInternal(unsigned a_NumLightBags, CDF* a_Cdf, LightBagEntry* a_LightBagPtr, TriangleLight* a_Lights, const std::uint32_t a_Seed)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < a_NumLightBags; i += stride)
    {
        //Generate a random float between 0 and 1.
        auto seed = a_Seed + i;
        float random = RandomFloat(seed);

        //Store the pdf and light in the light bag.
        unsigned lIndex;
        float pdf;
        a_Cdf->Get(random, lIndex, pdf);
        a_LightBagPtr[i] = {a_Lights[lIndex], pdf};
    }
}

__host__ void PickPrimarySamples(const WaveFront::RayData* const a_RayData, const WaveFront::IntersectionData* const a_IntersectionData, const LightBagEntry* const a_LightBags, Reservoir* a_Reservoirs, const ReSTIRSettings& a_Settings, PixelData* a_PixelData, const std::uint32_t a_Seed)
{
    //TODO ensure that each pixel grid operates within a single block, and that the L1 cache is not overwritten for each value. Optimize for cache hits.
    //TODO correctly assign a light bag per grid through some random generation.

    const auto numReservoirs = (a_Settings.width * a_Settings.height * a_Settings.numReservoirsPerPixel);
    const int blockSize = CUDA_BLOCK_SIZE;
    const int numBlocks = (numReservoirs + blockSize - 1) / blockSize;
    PickPrimarySamplesInternal<<<numBlocks, blockSize>>>(a_RayData, a_IntersectionData, a_LightBags, a_Reservoirs, a_Settings, a_PixelData, a_Seed);
    cudaDeviceSynchronize();
}

__global__ void PickPrimarySamplesInternal(const WaveFront::RayData* const a_RayData, const WaveFront::IntersectionData* const a_IntersectionData, const LightBagEntry* const a_LightBags, Reservoir* a_Reservoirs, const ReSTIRSettings& a_Settings, PixelData* a_PixelData, const std::uint32_t a_Seed)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const auto numPixels = a_Settings.width * a_Settings.height;

    //Seed for this thread index.
    auto lightBagSeed = a_Seed + blockIdx.x;    //Seed is the same for each block so that they all get the same light bag.
    float random = RandomFloat(lightBagSeed);

    //Generate between 0 and 1, then round and pick a light bag index based on the total light bag amount.
    int lightBagIndex = static_cast<int>(round(static_cast<float>(a_Settings.numLightBags - 1) * random));

    auto* pickedLightBag = &a_LightBags[lightBagIndex * a_Settings.numLightsPerBag];

    //Loop over the pixels
    for (int i = index; i < numPixels; i += stride)
    {
        //Get the intersection data for this pixel.
        auto* intersectionData = &a_IntersectionData[i];

        //Extract the pixel data from the right buffers, and store them for this pixel index.
        PixelData* pixel = &a_PixelData[i];

        //If no intersection exists at this pixel, do nothing.
        if(intersectionData->m_IntersectionT <= 0.f)
        {
            //Set pixel depth to 0 to be able to identify unused pixels easily.
            pixel->depth = -1;
            continue;
        }

        //The ray that resulted in this intersection.
        auto* ray = &a_RayData[intersectionData->m_RayArrayIndex];

        //Extract the pixel features from all the different buffers. Store them in the pixel.
        pixel->worldPosition = ray->m_Origin + ray->m_Direction * intersectionData->m_IntersectionT;
        pixel->directionIncoming = ray->m_Direction;

        const unsigned int vertexIndex = 3 * intersectionData->m_PrimitiveIndex;
        const DevicePrimitive* primitive = intersectionData->m_Primitive;
        const unsigned int vertexIndexA = primitive->m_IndexBuffer[vertexIndex + 0];
        const unsigned int vertexIndexB = primitive->m_IndexBuffer[vertexIndex + 1];
        const unsigned int vertexIndexC = primitive->m_IndexBuffer[vertexIndex + 2];

        const Vertex* A = &primitive->m_VertexBuffer[vertexIndexA];
        const Vertex* B = &primitive->m_VertexBuffer[vertexIndexB];
        const Vertex* C = &primitive->m_VertexBuffer[vertexIndexC];

        const float U = intersectionData->m_UVs.x;
        const float V = intersectionData->m_UVs.y;
        const float W = 1.f - (U + V);

        const float2 texCoords = A->m_UVCoord * W + B->m_UVCoord * U + C->m_UVCoord * V;
        const float4 texColor = tex2D<float4>(primitive->m_Material->m_DiffuseTexture, texCoords.x, texCoords.y);
        const float4 triangleColor = primitive->m_Material->m_DiffuseColor;

        pixel->worldNormal = normalize(A->m_Normal + B->m_Normal + C->m_Normal);
        pixel->diffuse = float3{texColor.x * triangleColor.x, texColor.y * triangleColor.y, texColor.z * triangleColor.z};
        pixel->roughness = 1.f; //intersectionData->m_Primitive->m_Material->m_Roughness; //TODO
        pixel->metallic = 0.f;  //intersectionData->m_Primitive->m_Material->m_Metallic; //TODO


        pixel->depth = intersectionData->m_IntersectionT;

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
                //Random number using the pixel id.
                auto seed = a_Seed + i;
                float r = RandomFloat(seed);

                const int pickedLightIndex = static_cast<int>(round(static_cast<float>(a_Settings.numLightsPerBag - 1) * r));
                const LightBagEntry pickedEntry = pickedLightBag[pickedLightIndex];
                const TriangleLight light = pickedEntry.light;
                const float initialPdf = pickedEntry.pdf;

                //Generate random UV coordinates. Between 0 and 1.
                const float u = RandomFloat(seed);  //Seed is altered after each shift, which makes it work with the same uint.
                const float v = RandomFloat(seed);

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
                    Resample(&lightSample, pixel, &lightSample);
                }

                //The final PDF for the light in this reservoir is the solid angle divided by the original PDF of the light being chosen based on radiance.
                const auto pdf = lightSample.solidAnglePdf / initialPdf;
                reservoir->Update(lightSample, pdf, a_Seed);
            }

            reservoir->UpdateWeight();
        }
    }
}

__host__ int GenerateReSTIRShadowRays(MemoryBuffer* a_AtomicCounter, Reservoir* a_Reservoirs, RestirShadowRay* a_ShadowRays, PixelData* a_PixelData)
{
    //Counter that is atomically incremented. Copy it to the GPU.
    int atomic = 0;
    a_AtomicCounter->Write(atomic);
    auto devicePtr = a_AtomicCounter->GetDevicePtr<int>();
    const auto numPixels = ReSTIRSettings::width * ReSTIRSettings::height;

    //Call in parallel.
    const int blockSize = CUDA_BLOCK_SIZE;
    const int numBlocks = (numPixels + blockSize - 1) / blockSize;
    GenerateShadowRay<<<numBlocks, blockSize>>> (devicePtr, a_Reservoirs, a_ShadowRays, a_PixelData);
    cudaDeviceSynchronize();

    //Copy value back to the CPU.
    a_AtomicCounter->Read(&atomic, sizeof(int), 0);

    //Return the amount of rays that have been generated.
    return atomic;
}

__global__ void GenerateShadowRay(int* a_AtomicCounter, Reservoir* a_Reservoirs, RestirShadowRay* a_ShadowRays, PixelData* a_PixelData)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const auto numPixels = ReSTIRSettings::width * ReSTIRSettings::height;

    for (int pixel = index; pixel < numPixels; pixel += stride)
    {
        PixelData* pixelData = &a_PixelData[pixel];        

        //Only run for valid intersections.
        if(pixelData->depth > 0.f)
        {
            float3 pixelPosition = pixelData->worldPosition;

            //Run for every reservoir for the pixel.
            for(int depth = 0; depth < ReSTIRSettings::numReservoirsPerPixel; ++depth)
            {
                //If the reservoir has a weight, add a shadow ray.
                Reservoir* reservoir = &a_Reservoirs[RESERVOIR_INDEX(pixel, depth, ReSTIRSettings::numReservoirsPerPixel)];
                if(reservoir->weight > 0.f)
                {

                    int shadowIndex = atomicAdd(a_AtomicCounter, 1);

                    float3 pixelToLight = (reservoir->sample.position - pixelPosition);
                    float l = length(pixelToLight);
                    pixelToLight /= l;

                    RestirShadowRay ray;
                    ray.index = pixel;
                    ray.direction = pixelToLight;
                    ray.origin = pixelPosition;
                    ray.distance = l - 0.005f; //Make length a little bit shorter to prevent self-shadowing.

                    a_ShadowRays[shadowIndex] = ray;
                }
            }            
        }
    }
}

__host__ void SpatialNeighbourSampling(Reservoir* a_Reservoirs, Reservoir* a_SwapBuffer, PixelData* a_PixelData, const std::uint32_t a_Seed)
{    
    const auto numPixels = (ReSTIRSettings::width * ReSTIRSettings::height);
    const int blockSize = CUDA_BLOCK_SIZE;
    const int numBlocks = (numPixels + blockSize - 1) / blockSize;
    SpatialNeighbourSamplingInternal<<<numBlocks, blockSize >>>(a_Reservoirs, a_SwapBuffer, a_PixelData, a_Seed);
    cudaDeviceSynchronize();
}

//TODO access settings struct statically instead of passing an instance.

__global__ void SpatialNeighbourSamplingInternal(Reservoir* a_Reservoirs, Reservoir* a_SwapBuffer,
    PixelData* a_PixelData, const std::uint32_t a_Seed)
{
    Reservoir* fromBuffer = a_Reservoirs;
    Reservoir* toBuffer = a_SwapBuffer;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const auto numPixels = (ReSTIRSettings::width * ReSTIRSettings::height);

    //Storage for reservoirs and pixels to be combined.
    PixelData* toCombinePixelData[ReSTIRSettings::numSpatialSamples + 1];
    Reservoir* toCombineReservoirs[ReSTIRSettings::numSpatialSamples + 1];

    //Loop over the pixels.
    for (int i = index; i < numPixels; i += stride)
    {
        //The seed unique to this pixel.
        auto seed = a_Seed + i;

        toCombinePixelData[0] = &a_PixelData[i];

        //Only run when there's an intersection for this pixel.
        if (toCombinePixelData[0]->depth > 0.f)
        {
            //TODO maybe store this information inside the pixel. Could calculate it once at the start of the frame.
            const int y = i / ReSTIRSettings::width;
            const int x = i - (y * ReSTIRSettings::width);

            for (int iteration = 0; iteration < ReSTIRSettings::numSpatialIterations; ++iteration)
            {
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
                        const int neighbourIndex = PIXEL_INDEX(neighbourX, neighbourY, ReSTIRSettings::numReservoirsPerPixel);

                        //Only run if within image bounds.
                        if(neighbourX >= 0 && neighbourX <= ReSTIRSettings::width && neighbourY >= 0 && neighbourY <= ReSTIRSettings::height)
                        {
                            Reservoir* pickedReservoir = &a_Reservoirs[RESERVOIR_INDEX(neighbourIndex, depth, ReSTIRSettings::numReservoirsPerPixel)];
                            PixelData* pickedPixel = &a_PixelData[neighbourIndex];

                            //Only run for valid depths.
                            if(pickedPixel->depth > 0)
                            {
                                //Gotta stay positive.
                                assert(pickedReservoir->weight >= 0.f);

                                //Discard samples that are too different.
                                float depth1 = pickedPixel->depth;
                                float depth2 = toCombinePixelData[0]->depth;
                                float depthDifPct = fabs(depth1 - depth2) / ((depth1 + depth2) / 2.f);

                                const float angleDif = dot(pickedPixel->worldNormal, toCombinePixelData[0]->worldNormal);	//Between 0 and 1 (0 to 90 degrees). 
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

                    //If valid reservoirs to combine were found, combine them.
                    if (count > 1)
                    {
                        if(ReSTIRSettings::enableBiased)
                        {
                            CombineBiased(i, count, toCombineReservoirs, toCombinePixelData, seed);
                        }
                        else
                        {
                            CombineUnbiased(i, count, toCombineReservoirs, toCombinePixelData, seed);
                        }
                    }
                }

                //Swap the pointers for in and output.
                Reservoir* temp = fromBuffer;
                fromBuffer = toBuffer;
                toBuffer = temp;
            }
        }
    }
}


__host__ void TemporalNeighbourSampling(
    Reservoir* a_CurrentReservoirs,
    Reservoir* a_PreviousReservoirs,
    PixelData* a_CurrentPixelData,
    PixelData* a_PreviousPixelData,
    const std::uint32_t a_Seed
)
{
    const int numPixels = (ReSTIRSettings::width * ReSTIRSettings::height);
    const int blockSize = CUDA_BLOCK_SIZE;
    const int numBlocks = (numPixels + blockSize - 1) / blockSize;

    //TODO pass the motion vector information in here.

    CombineTemporalSamplesInternal << <numBlocks, blockSize >> > (a_CurrentReservoirs, a_PreviousReservoirs,
                                                                  a_CurrentPixelData, a_PreviousPixelData, a_Seed);
    cudaDeviceSynchronize();
}


__global__ void CombineTemporalSamplesInternal(
    Reservoir* a_CurrentReservoirs,
    Reservoir* a_PreviousReservoirs,
    PixelData* a_CurrentPixelData,
    PixelData* a_PreviousPixelData,
    const std::uint32_t a_Seed
)
{
    const int numPixels = (ReSTIRSettings::width * ReSTIRSettings::height);
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    Reservoir* toCombine[2];
    PixelData* pixelPointers[2];

    for (int i = index; i < numPixels; i += stride)
    {
        //TODO instead look up the motion vector and use that to find the right pixel. This assumes a static scene rn.
        const int temporalIndex = i;

        pixelPointers[0] = &a_PreviousPixelData[temporalIndex];
        pixelPointers[1] = &a_CurrentPixelData[i];

        //Ensure that the depth of both samples is valid, and then combine them at each depth.
        if (pixelPointers[0]->depth > 0.f && pixelPointers[1]->depth > 0.f)
        {
            //For every reservoir at the current pixel.
            for (int depth = 0; depth < ReSTIRSettings::numReservoirsPerPixel; ++depth)
            {
                toCombine[0] = &a_PreviousReservoirs[RESERVOIR_INDEX(temporalIndex, depth, ReSTIRSettings::numReservoirsPerPixel)];
                toCombine[1] = &a_CurrentReservoirs[RESERVOIR_INDEX(i, depth, ReSTIRSettings::numReservoirsPerPixel)];

                //Discard samples that are too different.
                float depth1 = pixelPointers[0]->depth;
                float depth2 = pixelPointers[1]->depth;
                float depthDifPct = fabs(depth1 - depth2) / ((depth1 + depth2) / 2.f);

                const float angleDif = dot(pixelPointers[0]->worldNormal, pixelPointers[1]->worldNormal);	//Between 0 and 1 (0 to 90 degrees). 
                static constexpr float MAX_ANGLE_COS = 0.72222222223f;	//Dot product is cos of the angle. If higher than this value, it's within 25 degrees.

                //Only do something if the samples are not vastly different.
                if (depthDifPct < 0.10f && angleDif > MAX_ANGLE_COS)
                {
                    //Cap sample count at 20x current to reduce temporal influence. Would grow infinitely large otherwise.
                    toCombine[0]->sampleCount = fminf(toCombine[0]->sampleCount, toCombine[1]->sampleCount * 20);

                    if (ReSTIRSettings::enableBiased)
                    {
                        CombineBiased(i, 2, toCombine, pixelPointers, a_Seed + i);
                    }
                    else
                    {
                        CombineUnbiased(i, 2, toCombine, pixelPointers, a_Seed + i);
                    }
                }
                
            }
        }
    }
}

__device__ void CombineUnbiased(int a_PixelIndex, int a_Count, Reservoir** a_Reservoirs,
                                PixelData** a_ToCombine, const std::uint32_t a_Seed)
{

    for (int depth = 0; depth < ReSTIRSettings::numReservoirsPerPixel; ++depth)
    {
        Reservoir output;
        int sampleCountSum = 0;

        for (int index = 0; index < a_Count; ++index)
        {
            auto* otherReservoir = a_Reservoirs[RESERVOIR_INDEX(index, depth, ReSTIRSettings::numReservoirsPerPixel)];
            LightSample resampled;
            Resample(&otherReservoir->sample, a_ToCombine[index], &resampled);

            const float weight = static_cast<float>(otherReservoir->sampleCount) * otherReservoir->weight * resampled.
                solidAnglePdf;

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
                correction += a_Reservoirs[RESERVOIR_INDEX(otherPixel->index, depth, ReSTIRSettings::numReservoirsPerPixel)]->sampleCount;
            }
        }

        //TODO Shadow ray is shot here in ReSTIR to check visibility at every resampled pixel.


        //TODO I don't understand this part fully, but it's in the pseudocode of ReSTIR. Dive into it when I have time.
        const float m = 1.f / fmaxf(static_cast<float>(correction), MINFLOAT);
        output.weight = (1.f / fmaxf(output.sample.solidAnglePdf, MINFLOAT)) * (m * output.weightSum);

        //Store the output reservoir for the pixel.
        a_Reservoirs[RESERVOIR_INDEX(a_PixelIndex, depth, ReSTIRSettings::numReservoirsPerPixel)];
    }
}

__device__ void CombineBiased(int a_PixelIndex, int a_Count, Reservoir** a_Reservoirs,
                              PixelData** a_ToCombine, const std::uint32_t a_Seed)
{
    //Loop over every depth.
    for (int depth = 0; depth < ReSTIRSettings::numReservoirsPerPixel; ++depth)
    {
        Reservoir output;
        int sampleCountSum = 0;

        //Iterate over the intersection data to combine.
        for (int i = 0; i < a_Count; ++i)
        {
            auto* pixel = a_ToCombine[i];
            auto* reservoir = a_Reservoirs[RESERVOIR_INDEX(pixel->index, depth, ReSTIRSettings::numReservoirsPerPixel)];

            LightSample resampled;
            Resample(&reservoir->sample, pixel, &resampled);

            const float weight = static_cast<float>(reservoir->sampleCount) * reservoir->weight * resampled.
                solidAnglePdf;

            assert(resampled.solidAnglePdf >= 0.f);

            output.Update(resampled, weight, a_Seed);

            sampleCountSum += reservoir->sampleCount;
        }

        //Update the sample 
        output.sampleCount = sampleCountSum;
        output.UpdateWeight();

        assert(output.weight >= 0.f && output.weightSum >= 0.f);

        //Override the reservoir for the output at this depth.
        *a_Reservoirs[RESERVOIR_INDEX(a_PixelIndex, depth, ReSTIRSettings::numReservoirsPerPixel)] = output;
    }
}

__device__ void Resample(LightSample* a_Input, const PixelData* a_PixelData, LightSample* a_Output)
{
    *a_Output = *a_Input;

    float3 pixelToLightDir = a_Input->position - a_PixelData->worldPosition;
    //Direction from pixel to light.
    const float lDistance = length(pixelToLightDir);
    //Light distance from pixel.
    pixelToLightDir /= lDistance;
    //Normalize.
    const float cosIn = clamp(dot(pixelToLightDir, a_PixelData->worldNormal), 0.f, 1.f);
    //Lambertian term clamped between 0 and 1. SurfaceN dot ToLight
    const float cosOut = clamp(dot(a_Input->normal, -pixelToLightDir), 0.f, 1.f);
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
    const auto brdf = MicrofacetBRDF(-pixelToLightDir, a_PixelData->directionIncoming, a_PixelData->worldNormal,
                                     a_PixelData->diffuse, a_PixelData->metallic, a_PixelData->roughness);

    //The unshadowed contribution (contributed if no obstruction is between the light and surface) takes the BRDF,
    //geometry factor and solid angle into account. Also the light radiance.
    //The only thing missing from this is the scaling with the rest of the scene based on the reservoir PDF.
    const auto unshadowedPathContribution = brdf * solidAngle * cosIn * a_Output->radiance;
    a_Output->unshadowedPathContribution = unshadowedPathContribution;

    //For the PDF, I take the unshadowed path contribution as a single float value. Average for now.
    //TODO might be better to instead take the max value? Ask Jacco.
    a_Output->solidAnglePdf = (unshadowedPathContribution.x + unshadowedPathContribution.y + unshadowedPathContribution.
        z) / 3.f;
}

__host__ void GenerateWaveFrontShadowRays(Reservoir* a_Reservoirs, PixelData* a_PixelData, MemoryBuffer* a_Atomic,
    WaveFront::ShadowRayData* a_ShadowRays)
{
    const auto numPixels = ReSTIRSettings::width * ReSTIRSettings::height;
    //Call in parallel.
    const int blockSize = CUDA_BLOCK_SIZE;
    const int numBlocks = (numPixels + blockSize - 1) / blockSize;
    GenerateWaveFrontShadowRaysInternal<<<numBlocks, blockSize>>>(a_Reservoirs, a_PixelData, a_Atomic->GetDevicePtr<int>(), a_ShadowRays);
    cudaDeviceSynchronize();
}

__global__ void GenerateWaveFrontShadowRaysInternal(Reservoir* a_Reservoirs, PixelData* a_PixelData, int* a_Atomic, WaveFront::ShadowRayData* a_ShadowRays)
{
    const auto numPixels = ReSTIRSettings::width * ReSTIRSettings::height;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    //Temporary storage for the rays.
    WaveFront::ShadowRayData rays[ReSTIRSettings::numReservoirsPerPixel];

    for (int i = index; i < numPixels; i += stride)
    {
        PixelData* pixel = &a_PixelData[i];

        //Only generate shadow rays for pixels that hit a surface.
        if(pixel->depth > 0.f)
        {
            /*
             * TODO
             * Note: This currently divides the expected contribution per reservoir by the amount of reservoirs.
             * It's essentially like scaling down so that the total adds up to 100% if all shadow rays pass.
             * This does shoot one shadow ray per reservoir, but I think that's needed for accurate results.
             * If we are really desperate we could average the reservoir results and then send a single shadow ray.
             */

            for (int depth = 0; depth < ReSTIRSettings::numReservoirsPerPixel; ++depth)
            {
                //Get the contribution and scale it down based on the number of reservoirs.
                Reservoir* reservoir = &a_Reservoirs[RESERVOIR_INDEX(i, depth, ReSTIRSettings::numReservoirsPerPixel)];
                float3 contribution = (reservoir->sample.unshadowedPathContribution * (reservoir->weight / static_cast<float>(ReSTIRSettings::numReservoirsPerPixel)));

                //Generate a ray for this particular reservoir.
                int rayIndex = atomicAdd(a_Atomic, 1);
                float3 toLightDir = reservoir->sample.position - pixel->worldPosition;
                const float l = length(toLightDir);
                toLightDir /= l;

                //TODO ensure no shadow acne.
                a_ShadowRays[rayIndex] = WaveFront::ShadowRayData{pixel->worldPosition, toLightDir, l - 0.005f, contribution, WaveFront::ResultBuffer::OutputChannel::DIRECT};
            }
        }
    }
}

