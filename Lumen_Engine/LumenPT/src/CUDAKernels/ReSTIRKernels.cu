#include "ReSTIRKernels.cuh"

#include "../Shaders/CppCommon/RenderingUtility.h"
#include "../Shaders/CppCommon/WaveFrontDataStructs.h"
#include "../Shaders/CppCommon/Half4.h"
#include "../Shaders/CppCommon/Half2.h"
#include <cuda_runtime_api.h>
#include <cuda/device_atomic_functions.h>
#include <cassert>
#include <cmath>
#include "../Framework/Timer.h"
#include "disney.cuh"
#include "../Framework/CudaUtilities.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#define CUDA_BLOCK_SIZE 256

#define CUDA_BLOCK_SIZE_GRID dim3{32, 32, 1}

__host__ void ResetReservoirs(
    int a_NumReservoirs, 
    Reservoir* a_ReservoirPointer)
{
    CHECKLASTCUDAERROR;
    //Call in parallel.
    const int blockSize = CUDA_BLOCK_SIZE;
    const int numBlocks = (a_NumReservoirs + blockSize - 1) / blockSize;
    ResetReservoirInternal<<<numBlocks, blockSize>>>(a_NumReservoirs, a_ReservoirPointer);
	
    //TODO: Wait after every task may not be needed.Check if it is required between kernel calls.
    cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;
}

__global__ void ResetReservoirInternal(
    int a_NumReservoirs, 
    Reservoir* a_ReservoirPointer)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < a_NumReservoirs; i += stride)
    {
        a_ReservoirPointer[i].Reset();
    }
}

__host__ void FillCDF(
    CDF* a_Cdf, 
    float* a_CdfTreeBuffer, 
    WaveFront::AtomicBuffer<WaveFront::TriangleLight>* a_Lights, 
    unsigned a_LightCount)
{
	//Note: No need to manually reset as the CDF rebuilding manually sets the sum and size.
    //First reset the CDF on the GPU.
    //ResetCDF<<<1,1>>>(a_Cdf);
    //cudaDeviceSynchronize();
    //CHECKLASTCUDAERROR;

	//Note: Disabled because horribly slow.
    //Run from one thread because it's not thread safe to append the sum of each element.
    //FillCDFInternalSingleThread <<<1, 1>>> (a_Cdf, a_Lights, a_LightCount);


	//Sort the light buffer so that lights with low values are at the start.
	//This prevents floating point inaccuracies mounting up and completely nullifying light contributions.
	//By adding small lights first, their prefix sum in the CDF will have many significant bits where it matters.
	//First get the right GPU pointer offsets.
    char* lightsStart = reinterpret_cast<char*>(a_Lights) + (2 * sizeof(unsigned int)); //Atomic buffer contains two unsigned ints before the actual data.
    char* lightsEnd = lightsStart + (sizeof(WaveFront::TriangleLight) * a_LightCount); 
    thrust::device_ptr<WaveFront::TriangleLight> start(reinterpret_cast<WaveFront::TriangleLight*>(lightsStart));
    thrust::device_ptr<WaveFront::TriangleLight> end(reinterpret_cast<WaveFront::TriangleLight*>(lightsEnd));
    thrust::sort(start, end, TriangleLightComparator());
	


    /*
	 * Thrust offers a parallel prefix sum scan algorithm.
	 * First calculate the weights in the CDF, then in-place append.
	 */
     //Use 512 threads per block, since these are relatively tiny operations.
    const unsigned blockSize = 512;
    const unsigned blockCount = (a_LightCount + blockSize - 1) / blockSize;
    CalculateLightWeightsInCDF<<<blockCount, blockSize>>>(a_Cdf, a_Lights, a_LightCount);
    CHECKLASTCUDAERROR;
    thrust::device_ptr<float> cdfStart(reinterpret_cast<float*>(reinterpret_cast<char*>(a_Cdf) + (2 * sizeof(unsigned))));
    thrust::device_ptr<float> cdfEnd(cdfStart + a_LightCount);
    thrust::inclusive_scan(cdfStart, cdfEnd, cdfStart);
	
    //NOTE: Commented out because thrust offers a simpler version.
    ////Use 512 threads per block, since these are relatively tiny operations.
    //const unsigned blockSize = 512;
    //const unsigned treeDepth = std::ceil(std::log2f(a_LightCount)); //The total depth of the tree, in terms of operations to be performed.
    //const unsigned numLeafNodes = std::pow(2u, treeDepth);
    //const unsigned blockCount = (numLeafNodes + blockSize - 1) / blockSize;
	//Calculate the light weights and output to the tree buffer. Threads for lights that are no present will deposit 0 weight to keep the tree valid.
    //CalculateLightWeights << <blockCount, blockSize >> > (a_CdfTreeBuffer, a_Lights, a_LightCount, numLeafNodes);
    //CHECKLASTCUDAERROR;
    //cudaDeviceSynchronize();
	//
	////Offset into the array to start writing the tree.
 //   unsigned arrayOffset = numLeafNodes;
	//
	////Loop over each depth and build the tree.
	//for(auto depth = 0u; depth < treeDepth; ++depth)
	//{
 //       const unsigned numThreads = std::pow(2, treeDepth - (depth + 1));   //Half the amount of threads as there are lights at a depth.
 //       const int numBlocks = (numThreads + blockSize - 1) / blockSize;
 //       BuildCDFTree <<<numBlocks, blockSize >>> (a_CdfTreeBuffer, numThreads, arrayOffset - 1);  //numThreads is the amount of nodes to be written.
 //       arrayOffset /= 2;   //Half the offset, as half the amount of root nodes exist at the parent.
	//	CHECKLASTCUDAERROR;
 //       cudaDeviceSynchronize();
	//}

	////Spawn a thread per element in the CDF. Fill by traversing down tree on the left.
 //   FillCDFParallel <<<((a_LightCount + blockSize - 1) / blockSize), blockSize >>> (a_Cdf, a_CdfTreeBuffer, a_Lights, a_LightCount, treeDepth, numLeafNodes);
 //   cudaDeviceSynchronize();
 //   CHECKLASTCUDAERROR;
	
	//Set the CDF to the right size.
    SetCDFSize<<<1, 1>>>(a_Cdf, a_LightCount);
	
    cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;

	//TODO remove
    //DebugPrintCdf<<<1,1>>>(a_Cdf, a_CdfTreeBuffer);

}

__global__ void ResetCDF(CDF* a_Cdf)
{
    a_Cdf->Reset();
}

__global__ void CalculateLightWeights(
    float* a_CdfTreeBuffer, 
    const WaveFront::AtomicBuffer<WaveFront::TriangleLight>* a_Lights, 
    unsigned a_LightCount, 
    unsigned a_NumLeafNodes)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    //Loop over all the light indices assigned to this thread.
    for (int i = index; i < a_NumLeafNodes; i += stride)
    {
        float weight = 0.f;

    	//If there is a light with this index, calculate the weight.
        if(i < a_LightCount)
        {
            //Light weight is just the average radiance for now.
            const float3 radiance = a_Lights->GetData(i)->radiance;
            assert(radiance.x >= 0.f && radiance.y >= 0.f && radiance.z >= 0.f && "Radiance needs to be positive, no taking away the light in the soul.");
            weight = (radiance.x + radiance.y + radiance.z) / 3.f;
        }

    	//Output to the END of the buffer so that the tree can be built on top with root = 0.
        a_CdfTreeBuffer[i + a_NumLeafNodes - 1] = weight;
    }
}

__global__ void CalculateLightWeightsInCDF(
    CDF* a_Cdf, 
    const WaveFront::AtomicBuffer<WaveFront::TriangleLight>* a_Lights, 
    unsigned a_LightCount)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    //Loop over all the light indices assigned to this thread.
    for (int i = index; i < a_LightCount; i += stride)
    {
        //Light weight is just the average radiance for now.
        const float3 radiance = a_Lights->GetData(i)->radiance;
        assert(radiance.x >= 0.f && radiance.y >= 0.f && radiance.z >= 0.f && "Radiance needs to be positive, no taking away the light in the soul.");

        //Output to the END of the buffer so that the tree can be built on top with root = 0.
        a_Cdf->data[i] = (radiance.x + radiance.y + radiance.z) / 3.f;
    }
}

__global__ void SetCDFSize(
    CDF* a_Cdf, 
    unsigned a_NumLights)
{
    a_Cdf->SetCDFSize(a_NumLights);
}

__global__ void DebugPrintCdf(
    CDF* a_Cdf, 
    float* a_CDFTree)
{
    //int start = a_Cdf->size - 30;
    //if (start < 0) start = 0;

    //const unsigned treeSize = powf(2.f, ceilf(log2f(a_Cdf->size)));

    //printf("CDF Entry 0: %f\n", a_Cdf->data[0]);
    //printf("CDF Tree Entry 0: %f\n", a_CDFTree[treeSize - 1]);
	
    //for (int i = start; i < a_Cdf->size; ++i)
    //{
    //    printf("CDF Entry %i: %f\n", i, a_Cdf->data[i]);
    //}

    //for (int i = 0; i < min(30u, a_Cdf->size - 1); ++i)
    //{
    //    printf("CDF tree node %i: %f\n", i, a_CDFTree[i]);
    //}

	//Check CDF validity.
    float prev = 0.f;
	for(int i = 0; i < a_Cdf->size; ++i)
	{
        const float current = a_Cdf->data[i];
		if(current - prev < EPSILON)
		{
            printf("CDF entry %i is the same as %i. Value: %f.\n", i, i - 1, current);
		}
        prev = current;
		
	}
}


__global__ void BuildCDFTree(
    float* a_CdfTreeBuffer, 
    unsigned a_NumParentNodes, 
    unsigned a_ArrayOffset)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
	
	//Loop over all the light indices assigned to this thread.
    for (int i = index; i < a_NumParentNodes; i += stride)
    {
        //Combine the previous two elements into a single new element, and output it in the right position.
        const int childRoot = a_ArrayOffset + (2 * i);
        const int parentRoot = (childRoot - 1) / 2;
        const float sum = a_CdfTreeBuffer[childRoot] + a_CdfTreeBuffer[childRoot + 1];
        a_CdfTreeBuffer[parentRoot] = sum;
	}
}

__global__ void FillCDFParallel(
    CDF* a_Cdf, 
    float* a_CdfTreeBuffer, 
    const WaveFront::AtomicBuffer<WaveFront::TriangleLight>* a_Lights, 
    unsigned a_LightCount, 
    unsigned a_TreeDepth, 
    unsigned a_LeafNodes)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

	//For every light, traverse the tree and find the sum.
    for (int lightIndex = index; lightIndex < a_LightCount; lightIndex += stride)
    {
        unsigned start = 0;
        unsigned end = a_LeafNodes - 1;
        unsigned branchIndex = 0;
    	
        /*
         * The appended value found by traversing the tree.
         * When the tree reaches the bottom level, it terminates, and so the element at light index
         * is not actually added.
         * to solve this, start by reading the bottom element derived from the light index.
         */
        float sum = a_CdfTreeBuffer[(a_LeafNodes - 1) + lightIndex];    
    	
        for(int depth = 0; depth < a_TreeDepth; ++depth)
        {
        	//Center of the nodes remaining.
            const int split = (start + end) / 2;

        	//Light index lies left of the split, so the right nodes are fully discarded.
        	if(lightIndex <= split)
        	{        		
        		//Go down the tree on the left.
                branchIndex = (2 * branchIndex) + 1;
        		
        		//Adjust the search range to the left.
                end = split;
        	}
        	//Light index lies in right nodes. Append left and traverse down the right side.
            else
            {
                //Go down the tree on the right. Append the found 
                const auto leftIndex = (2 * branchIndex) + 1;
                sum += a_CdfTreeBuffer[leftIndex];
                branchIndex = leftIndex + 1;  //Right node index
            	
            	//Adjust the search region to the right.
                start = split + 1;
            }
        }

        //Finally add to the CDF when all branches relevant for the index have been appended.
        a_Cdf->Insert(lightIndex, sum);
    }
}

__global__ void FillCDFInternalSingleThread(
    CDF* a_Cdf, 
    const WaveFront::AtomicBuffer<WaveFront::TriangleLight>* a_Lights, 
    unsigned a_LightCount)
{
    for (int i = 0; i < a_LightCount; ++i)
    {
        //Weight is the average illumination for now. Could take camera into account.
        const float3 radiance = a_Lights->GetData(i)->radiance;

        //printf("Radiance: %f, %f, %f LightCount: %i \n", radiance.x, radiance.y, radiance.z, a_LightCount);

        assert(radiance.x >= 0.f && radiance.y >= 0.f && radiance.z >= 0.f && "Radiance needs to be positive, no taking away the light in the soul.");

        const float weight = (radiance.x + radiance.y + radiance.z) / 3.f;
        assert(weight >= 0.f);
    	
        a_Cdf->Insert(weight);
    }
}

__host__ void FillLightBags(
    unsigned a_NumLightBags, 
    unsigned a_NumLightsPerBag, 
    CDF* a_Cdf, 
    LightBagEntry* a_LightBagPtr, 
    const WaveFront::AtomicBuffer<WaveFront::TriangleLight>* a_Lights, 
    const std::uint32_t a_Seed)
{
    const unsigned numLightsTotal = a_NumLightBags * a_NumLightsPerBag;
    const int blockSize = CUDA_BLOCK_SIZE;
    const int numBlocks = (numLightsTotal + blockSize - 1) / blockSize;
    FillLightBagsInternal <<<numBlocks, blockSize >>>(a_NumLightBags, a_NumLightsPerBag, a_Cdf, a_LightBagPtr, a_Lights, a_Seed);
    cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;
}

__global__ void FillLightBagsInternal(
    unsigned a_NumLightBags, 
    unsigned a_NumLightsPerBag, 
    CDF* a_Cdf, 
    LightBagEntry* a_LightBagPtr, 
    const WaveFront::AtomicBuffer<WaveFront::TriangleLight>* a_Lights, 
    const std::uint32_t a_Seed)
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
    	
        assert(pdf >= 0.f);
    	
        a_LightBagPtr[i] = LightBagEntry{*a_Lights->GetData(lIndex), pdf};
    }
}

__host__ void PickPrimarySamples(
    const LightBagEntry* const a_LightBags, 
    Reservoir* a_Reservoirs, 
    const ReSTIRSettings& a_Settings, 
    const WaveFront::SurfaceData * const a_PixelData, 
    const std::uint32_t a_Seed)
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
        a_Settings.numLightBags,
        a_Settings.numLightsPerBag,
        a_PixelData,
        a_Seed
    );
    cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;
}

__global__ void PickPrimarySamplesInternal(
    const LightBagEntry* const a_LightBags, 
    Reservoir* a_Reservoirs, 
    unsigned a_NumPrimarySamples, 
    unsigned a_NumReservoirs, 
    unsigned a_NumLightBags, 
    unsigned a_NumLightsPerBag, 
    const WaveFront::SurfaceData * 
    const a_PixelData, 
    const std::uint32_t a_Seed)
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
    //TODO: Note: These artifacts seem to appear but only on the first few frames. With multiple reservoirs per pixel it resolves itself.
    //TODO: It's only really noticeable when there is many lights of similar size to choose from, and light bags can be full of bad samples. Or small light bags.
    //TODO: Conclusion: it's fine unless light bag to light ratio is too big. In those cases interleaved or per block doesn't really matter.
    //TODO: So in those cases instead try to have a separate light bag per reservoir: seed + smid + depth. But then you lose the cache coherency somewhat.
    auto lBagSeed = WangHash(a_Seed + __mysmid());

    const float random = RandomFloat(lBagSeed);

    //Generate between 0 and 1, then round and pick a light bag index based on the total light bag amount.
    const int lightBagIndex = static_cast<int>(roundf(static_cast<float>(a_NumLightBags - 1) * random));
    auto* pickedLightBag = &a_LightBags[lightBagIndex * a_NumLightsPerBag];

    const auto pixelIndex = index / ReSTIRSettings::numReservoirsPerPixel;

    //The current pixel index.
    const WaveFront::SurfaceData pixel = a_PixelData[pixelIndex];

    //If the surface is not a valid intersection, set its weight to 0 to prevent reuse.
    if(pixel.m_SurfaceFlags)
    {
        a_Reservoirs[index].weight = 0.f;
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
            Resample(&lightSample, &pixel, &lightSample);
        }

        //The final PDF for the light in this reservoir is the solid angle divided by the original PDF of the light being chosen based on radiance.
        //Dividing scales the value up.
        const auto pdf = lightSample.solidAnglePdf / initialPdf;

        assert(!isnan(lightSample.solidAnglePdf));
        assert(!isinf(lightSample.solidAnglePdf));
        assert(lightSample.solidAnglePdf >= 0.f);
    	
        assert(!isnan(initialPdf));
        assert(!isinf(initialPdf));
        assert(initialPdf > 0.f);
    	
        assert(!isnan(pdf));
        assert(!isinf(pdf));
        assert(pdf >= 0.f);
    	
        fresh.Update(lightSample, pdf, seed);
    }

    //Finally update the reservoir weight.
    fresh.UpdateWeight();

    //Override the old reservoir.
    auto* reservoir = &a_Reservoirs[index];
    *reservoir = fresh;
}

__host__ unsigned int GenerateReSTIRShadowRays(
    MemoryBuffer* a_AtomicBuffer, 
    Reservoir* a_Reservoirs,
    const WaveFront::SurfaceData* a_PixelData, 
    unsigned a_NumReservoirs)
{
    //Counter that is atomically incremented. Copy it to the GPU.
    WaveFront::ResetAtomicBuffer<RestirShadowRay>(a_AtomicBuffer);

    const auto devicePtr = a_AtomicBuffer->GetDevicePtr<WaveFront::AtomicBuffer<RestirShadowRay>>();

    //Call in parallel.
    const int blockSize = CUDA_BLOCK_SIZE;
    const int numBlocks = (a_NumReservoirs + blockSize - 1) / blockSize;
    GenerateShadowRay<<<numBlocks, blockSize>>> (devicePtr, a_Reservoirs, a_PixelData, a_NumReservoirs);
    cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;

    //Copy value back to the CPU.
    return WaveFront::GetAtomicCounter<RestirShadowRay>(a_AtomicBuffer);
}

__global__ void GenerateShadowRay(
    WaveFront::AtomicBuffer<RestirShadowRay>* a_AtomicBuffer, 
    Reservoir* a_Reservoirs,
    const WaveFront::SurfaceData* a_PixelData, 
    unsigned a_NumReservoirs)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= a_NumReservoirs) return;

    const auto pixelIndex = index / ReSTIRSettings::numReservoirsPerPixel;
    const WaveFront::SurfaceData pixelData = a_PixelData[pixelIndex];

    //Only run for valid intersections.
    if(!pixelData.m_SurfaceFlags)
    {
        //If the reservoir has a weight, add a shadow ray.
        const Reservoir reservoir = a_Reservoirs[index];

        if(reservoir.weight > 0.f)
        {
            float3 pixelToLight = (reservoir.sample.position - pixelData.m_Position);
            const float l = length(pixelToLight);
            pixelToLight /= l;

            assert(fabsf(length(pixelToLight) - 1.f) <= FLT_EPSILON * 5.f);

            RestirShadowRay ray;
            ray.index = index;
            ray.direction = pixelToLight;
            ray.origin = pixelData.m_Position;
            ray.distance = l - 0.05f; //Make length a little bit shorter to prevent self-shadowing.
        	
            //TODO: this is a slow operation. Perhaps it's better to create multiple shadow rays per thread, store them locally, then add them at once?
            a_AtomicBuffer->Add(&ray);
        }         
    }
}

__host__ void Shade(
    Reservoir* a_Reservoirs, 
    unsigned a_Width, 
    unsigned a_Height, 
    cudaSurfaceObject_t a_OutputBuffer)
{
    const dim3 blockSize = CUDA_BLOCK_SIZE_GRID;
    const unsigned gridWidth = static_cast<unsigned>(std::ceil(static_cast<float>(a_Width) / static_cast<float>(blockSize.x)));
    const unsigned gridHeight = static_cast<unsigned>(std::ceil(static_cast<float>(a_Height) / static_cast<float>(blockSize.y)));
    const dim3 numBlocks {gridWidth, gridHeight, 1};

    ShadeInternal << <numBlocks, blockSize >> > (a_Reservoirs, a_Width, a_Height, a_OutputBuffer);
    cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;
}

__global__ void ShadeInternal(
    Reservoir* a_Reservoirs, 
    unsigned a_Width, 
    unsigned a_Height, 
    cudaSurfaceObject_t a_OutputBuffer)
{
    const unsigned int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    //Make sure to not go out of bounds.
    if(pixelX >= a_Width || pixelY >= a_Height)
    {
        return;
    }

	//Inlined shading function to shade this pixel at every depth in this reservoir set.
    ShadeReservoirs(a_Reservoirs, a_Width, pixelX, pixelY, pixelX, pixelY, a_OutputBuffer);
}

__device__ __forceinline__ void ShadeReservoirs(
    Reservoir* a_Reservoirs, 
    unsigned a_Width, 
    unsigned a_InputX, 
    unsigned a_InputY, 
    unsigned a_OutputX, 
    unsigned a_OutputY, 
    cudaSurfaceObject_t a_OutputBuffer)
{
    //The amount of samples shaded per pixel. Compile time constant. Used to scale contributions back down.
    constexpr auto numShadedSamples = ReSTIRSettings::numReservoirsPerPixel * (1 + (ReSTIRSettings::enableTemporal ? 1 : 0) + (ReSTIRSettings::enableSpatial ? 1 : 0));
	
    //Read the current shading value in the output buffer.
    half4Ushort4 color{ 0 };

    surf2Dread<ushort4>(
        &color.m_Ushort4,
        a_OutputBuffer,
        a_OutputX * sizeof(ushort4),
        a_OutputY,
        cudaBoundaryModeTrap);

	//The index of the pixel in terms of reservoirs.
    const auto pixelDataIndex = PIXEL_DATA_INDEX(a_InputX, a_InputY, a_Width);
	
    //Extract and scale reservoir shading values. Append to color.
    for (int depth = 0; depth < ReSTIRSettings::numReservoirsPerPixel; ++depth)
    {
        //Local copy of the reservoir in question.
        auto& reservoir = a_Reservoirs[RESERVOIR_INDEX(pixelDataIndex, depth, ReSTIRSettings::numReservoirsPerPixel)];

        //If the reservoir has a weight, append it to the shading.
        if (reservoir.weight > 0.f)
        {
            //Take the average contribution scaled after all reservoirs.
            color.m_Half4 += half4(make_float4(reservoir.sample.unshadowedPathContribution * (reservoir.weight / static_cast<float>(numShadedSamples)), 0.f));
        }
    }

    //Write the combined values to the output buffer.
    surf2Dwrite<ushort4>(
        color.m_Ushort4,
        a_OutputBuffer,
        a_OutputX * sizeof(ushort4),
        a_OutputY,
        cudaBoundaryModeTrap);
}

///*
// * Generate the shadow rays used for final shading.
// */
//__host__ unsigned int GenerateReSTIRShadowRaysShading(MemoryBuffer* a_AtomicBuffer, Reservoir* a_Reservoirs, const WaveFront::SurfaceData* a_PixelData, uint2 a_Resolution)
//{
//    //Counter that is atomically incremented. Copy it to the GPU.
//    WaveFront::ResetAtomicBuffer<RestirShadowRayShading>(a_AtomicBuffer);
//
//    const auto devicePtr = a_AtomicBuffer->GetDevicePtr<WaveFront::AtomicBuffer<RestirShadowRayShading>>();
//
//    //Call in parallel.
//    const dim3 blockSize = CUDA_BLOCK_SIZE_GRID;
//
//    const unsigned gridWidth = static_cast<unsigned>(std::ceil(static_cast<float>(a_Resolution.x) / static_cast<float>(blockSize.x)));
//    const unsigned gridHeight = static_cast<unsigned>(std::ceil(static_cast<float>(a_Resolution.y) / static_cast<float>(blockSize.y)));
//
//    const dim3 numBlocks {gridWidth, gridHeight, 1};
//
//    for (int depth = 0; depth < ReSTIRSettings::numReservoirsPerPixel; ++depth)
//    {
//        GenerateShadowRayShading <<<numBlocks, blockSize >>> (devicePtr, a_Reservoirs, a_PixelData, a_Resolution, depth);
//    }
//    cudaDeviceSynchronize();
//
//    CHECKLASTCUDAERROR;
//
//    //Copy value back to the CPU.
//    return WaveFront::GetAtomicCounter<RestirShadowRayShading>(a_AtomicBuffer);
//}
//
//__global__ void GenerateShadowRayShading(WaveFront::AtomicBuffer<RestirShadowRayShading>* a_AtomicBuffer, Reservoir* a_Reservoirs, const WaveFront::SurfaceData* a_PixelData, uint2 a_Resolution, unsigned a_Depth)
//{
//
//    const unsigned int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
//    const unsigned int pixelY = blockIdx.y * blockDim.y + threadIdx.y;
//
//    //Make sure to not go out of bounds.
//    if(pixelX >= a_Resolution.x || pixelY >= a_Resolution.y)
//    {
//        return;
//    }
//
//    const unsigned int pixelDataIndex = PIXEL_DATA_INDEX(pixelX, pixelY, a_Resolution.x);
//
//    const WaveFront::SurfaceData pixelData = a_PixelData[pixelDataIndex];
//
//    //Only run for valid intersections.
//    if (!pixelData.m_SurfaceFlags)
//    {
//        //If the reservoir has a weight, add a shadow ray.
//        const auto reservoirIndex = RESERVOIR_INDEX(pixelDataIndex, a_Depth, ReSTIRSettings::numReservoirsPerPixel);
//        const Reservoir reservoir = a_Reservoirs[reservoirIndex];
//
//        if (reservoir.weight > 0.f)
//        {
//            float3 pixelToLight = (reservoir.sample.position - pixelData.m_Position);
//            const float l = length(pixelToLight);
//            pixelToLight /= l;
//
//            assert(fabsf(length(pixelToLight) - 1.f) <= FLT_EPSILON * 5.f);
//
//            RestirShadowRayShading ray;
//            ray.index = reservoirIndex;
//            ray.pixelIndex = { pixelX, pixelY };
//            ray.direction = pixelToLight;
//            ray.origin = pixelData.m_Position;
//            ray.distance = l - 0.05f; //Make length a little bit shorter to prevent self-shadowing.
//
//            //Take the average contribution scaled after all reservoirs.
//            ray.contribution = (reservoir.sample.unshadowedPathContribution * (reservoir.weight / static_cast<float>(ReSTIRSettings::numReservoirsPerPixel)));
//        	
//            //TODO: this is a slow operation. Perhaps it's better to create multiple shadow rays per thread, store them locally, then add them at once?
//            a_AtomicBuffer->Add(&ray);
//        }
//    }
//}


__host__ Reservoir* SpatialNeighbourSampling(
    Reservoir* a_InputReservoirs,
    Reservoir* a_SwapBuffer1,
    Reservoir* a_SwapBuffer2,
    const WaveFront::SurfaceData* a_PixelData,
    const std::uint32_t a_Seed,
    uint2 a_Dimensions
)
{
    const unsigned numReservoirs = a_Dimensions.x * a_Dimensions.y * ReSTIRSettings::numReservoirsPerPixel;
    const int blockSize = CUDA_BLOCK_SIZE;
    const int numBlocks = (numReservoirs + blockSize - 1) / blockSize;

	//Initially, copy from the current reservoirs. After depth 0 the two swap buffers are used.
    Reservoir* fromBuffer = a_InputReservoirs;
    Reservoir* toBuffer = a_SwapBuffer1;

    //Synchronize between each swap, and then swap the buffers.
    for (int iteration = 0; iteration < ReSTIRSettings::numSpatialIterations; ++iteration)
    {
        SpatialNeighbourSamplingInternal<<<numBlocks, blockSize >>>(fromBuffer, toBuffer, a_PixelData, a_Seed, a_Dimensions, numReservoirs);
        cudaDeviceSynchronize();
        CHECKLASTCUDAERROR;

        //Swap the pointers for in and output.
    	if(iteration == 0)
    	{
            fromBuffer = a_SwapBuffer1;
            toBuffer = a_SwapBuffer2;
    	}
        else
        {
            Reservoir* temp = fromBuffer;
            fromBuffer = toBuffer;
            toBuffer = temp;
        }
    }

	//Buffer currently containing the combined reservoirs.
    return fromBuffer;
}

__global__ void SpatialNeighbourSamplingInternal(
    Reservoir* a_ReservoirsIn,
    Reservoir* a_ReservoirsOut,
    const WaveFront::SurfaceData* a_PixelData,
    const std::uint32_t a_Seed,
    uint2 a_Dimensions,
    unsigned a_NumReservoirs
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= a_NumReservoirs) return;

    //Storage for reservoirs and pixels to be combined.
    const WaveFront::SurfaceData* toCombinePixelData[ReSTIRSettings::numSpatialSamples];
    Reservoir* toCombineReservoirs[ReSTIRSettings::numSpatialSamples];

    //The seed unique to this pixel.
    auto seed =  WangHash(a_Seed + index);

    const auto pixelIndex = index / ReSTIRSettings::numReservoirsPerPixel;
    const auto currentDepth = index - (pixelIndex * ReSTIRSettings::numReservoirsPerPixel);

	//Current surface data.
    auto& currentSurfaceData = a_PixelData[pixelIndex];

    //Only run when there's an intersection for this pixel.
    if (!currentSurfaceData.m_SurfaceFlags)
    {
        const int y = pixelIndex / a_Dimensions.x;
        const int x = pixelIndex - (y * a_Dimensions.x);
        int count = 0;

        for (int neighbour = 0; neighbour < ReSTIRSettings::numSpatialSamples; ++neighbour)
        {
            //TODO This generates a square rn. Make it within a circle.
            const int neighbourY = round((RandomFloat(seed) * 2.f - 1.f) * static_cast<float>(ReSTIRSettings::spatialSampleRadius)) + y;
            const int neighbourX = round((RandomFloat(seed) * 2.f - 1.f) * static_cast<float>(ReSTIRSettings::spatialSampleRadius)) + x;

            //Only run if within image bounds.
            if (neighbourX >= 0 && neighbourX < a_Dimensions.x && neighbourY >= 0 && neighbourY < a_Dimensions.y)
            {
                //Index of the neighbouring pixel in the pixel array.
                const int neighbourIndex = PIXEL_INDEX(neighbourX, neighbourY, a_Dimensions.x);

                //Ensure no out of bounds neighbours are selected.
                assert(neighbourIndex < a_Dimensions.x * a_Dimensions.y && neighbourIndex >= 0);

                //Overwrite in the array, but don't up the counter yet.
                toCombinePixelData[count] = &a_PixelData[neighbourIndex];

                //Only run for valid depths and non-emissive surfaces.
                if (!toCombinePixelData[count]->m_SurfaceFlags)
                {
                    toCombineReservoirs[count] = &a_ReservoirsIn[RESERVOIR_INDEX(neighbourIndex, currentDepth, ReSTIRSettings::numReservoirsPerPixel)];
                    //Gotta stay positive.
                    assert(toCombineReservoirs[count]->weight >= 0.f);

                    //Discard samples that are too different.
                    const float depth1 = toCombinePixelData[count]->m_IntersectionT;
                    const float depth2 = currentSurfaceData.m_IntersectionT;
                    const float depthDifPct = fabs(depth1 - depth2) / ((depth1 + depth2) / 2.f);

                    const float angleDif = dot(toCombinePixelData[count]->m_Normal, currentSurfaceData.m_Normal);	//Between 0 and 1 (0 to 90 degrees). 
                    static constexpr float MAX_ANGLE_COS = 0.72222222223f;	//Dot product is cos of the angle. If higher than this value, it's within 25 degrees.

                    //If the samples are similar enough, up the counter. This will means the samples are not overwritten and will be merged.
                    if (depthDifPct < 0.10f && angleDif > MAX_ANGLE_COS)
                    {
                        ++count;
                    }
                }
            }
        }

        //If valid reservoirs to combine were found, combine them.
        if (count > 1)
        {
            //INLINE reservoir sampling for efficiency.
            if (ReSTIRSettings::enableBiased)
            {
                long long int sampleCountSum = 0;
                //Output reservoir.
                Reservoir output;

                //Iterate over the intersection data to combine.
                for (int i = 0; i < count; ++i)
                {
                    auto* reservoir = toCombineReservoirs[i];
                    LightSample resampled;
                    Resample(&(reservoir->sample), toCombinePixelData[0], &resampled);
                    const float weight = static_cast<float>(reservoir->sampleCount) * reservoir->weight * resampled.solidAnglePdf;

                    assert(!isnan(weight));
                    assert(!isinf(weight));
                    assert(weight >= 0.f);
                	
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
                a_ReservoirsOut[index] = output;
            }
            else
            {
                //Ensure enough reservoirs are passed.
                Reservoir output;
                int sampleCountSum = 0;

                //Merge the other reservoirs in.
                for (int i = 0; i < count; ++i)
                {
                    auto* otherReservoir = toCombineReservoirs[i];
                    LightSample resampled;
                    Resample(&otherReservoir->sample, toCombinePixelData[0], &resampled);

                    assert(otherReservoir->weight >= 0.f);
                    assert(otherReservoir->sampleCount >= 0);
                    assert(otherReservoir->weightSum >= 0.f);
                    assert(otherReservoir->sample.solidAnglePdf >= 0.f);

                    const float weight = static_cast<float>(otherReservoir->sampleCount) * otherReservoir->weight * resampled.solidAnglePdf;

                    assert(!isnan(weight));
                    assert(!isinf(weight));
                    assert(weight >= 0.f);
                	
                    output.Update(resampled, weight, a_Seed);

                    sampleCountSum += otherReservoir->sampleCount;
                }

                output.sampleCount = sampleCountSum;

                //Weigh against other pixels to remove bias from their solid angle by re-sampling.
                int correction = 0;

                for (int i = 0; i < count; ++i)
                {
                    auto* otherPixel= toCombinePixelData[i];
                    LightSample resampled;
                    Resample(&output.sample, otherPixel, &resampled);

                    if (resampled.solidAnglePdf > 0)
                    {
                        correction += a_ReservoirsOut[index].sampleCount;
                    }
                }


                //TODO Shadow ray is shot here in ReSTIR to check visibility at every resampled pixel.

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
                a_ReservoirsOut[index] = output;
            }
        }
        //Not enough reservoirs to combine, but still data needs to be passed on.
        else
        {
            //Set the output reservoir to 0 if there was no valid candidates. Will be auto discarded when combined for next frame.
            a_ReservoirsOut[index].Reset();
        }
    }
    
}


__host__ void TemporalNeighbourSampling(
    Reservoir* a_CurrentReservoirs,
    Reservoir* a_PreviousReservoirs,
    const WaveFront::SurfaceData* a_CurrentPixelData,
    const WaveFront::SurfaceData* a_PreviousPixelData,
    const std::uint32_t a_Seed,
    uint2 a_Dimensions,
    const cudaSurfaceObject_t a_MotionVectorBuffer,
    cudaSurfaceObject_t a_OutputBuffer
)
{
    const unsigned numPixels = a_Dimensions.x * a_Dimensions.y;
    const int blockSize = CUDA_BLOCK_SIZE;
    const int numBlocks = (numPixels + blockSize - 1) / blockSize;

    CombineTemporalSamplesInternal <<<numBlocks, blockSize >>> (
        a_CurrentReservoirs, 
        a_PreviousReservoirs,
        a_CurrentPixelData, 
        a_PreviousPixelData, 
        a_Seed, 
        numPixels,
        a_Dimensions, 
        a_MotionVectorBuffer,
        a_OutputBuffer
        );

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
    const cudaSurfaceObject_t a_MotionVectorBuffer,
    cudaSurfaceObject_t a_OutputBuffer
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= a_NumPixels) return;


    const int currentPixelY = index / a_Dimensions.x;
    const int currentPixelX = index - (currentPixelY * a_Dimensions.x);

	//Get the motion vector and X,Y coordinates from the current index (one index per pixel).
    half2Ushort2 motionVector = {half2{0.f, 0.f}};

    surf2Dread<ushort2>(
        &motionVector.m_Ushort2,
        a_MotionVectorBuffer,
        currentPixelX * sizeof(ushort2),
        currentPixelY,
        cudaBoundaryModeTrap);

    const auto velocity = motionVector.AsFloat2();
    const int movedX = roundf(static_cast<float>(a_Dimensions.x) * velocity.x);
    const int movedY = roundf(static_cast<float>(a_Dimensions.y) * velocity.y);
    int temporalPixelY = currentPixelY + movedY;
    int temporalPixelX = currentPixelX + movedX;
	
    //Set to current pixel index so that invalid values will just try to match the same pixel in the last frame. Who knows if it works?
    int temporalIndex = index;
    if (temporalPixelY >= 0 && temporalPixelY < a_Dimensions.y && temporalPixelX >= 0 && temporalPixelX < a_Dimensions.x)
    {
        temporalIndex = PIXEL_INDEX(temporalPixelX, temporalPixelY, a_Dimensions.x);
    }
    else
    {
    	//Set to current coords when out of bounds.
        temporalPixelX = currentPixelX;
        temporalPixelY = currentPixelY;
    }

    assert(temporalIndex >= 0 && temporalIndex < a_NumPixels);
    assert(index >= 0 && index < a_NumPixels);

    /*
     * Combine the reservoirs for every depth.
     * Also perform the shading.
     */
	
    for (int currentDepth = 0; currentDepth < ReSTIRSettings::numReservoirsPerPixel; ++currentDepth)
    {
        Reservoir toCombine[2];
        WaveFront::SurfaceData pixelPointers[2];
        const auto reservoirIndex = RESERVOIR_INDEX(index, currentDepth, ReSTIRSettings::numReservoirsPerPixel);
        const auto temporalReservoirIndex = RESERVOIR_INDEX(temporalIndex, currentDepth, ReSTIRSettings::numReservoirsPerPixel);

        assert(reservoirIndex >= 0 && reservoirIndex < a_NumPixels* ReSTIRSettings::numReservoirsPerPixel);
        assert(temporalReservoirIndex >= 0 && temporalReservoirIndex < a_NumPixels* ReSTIRSettings::numReservoirsPerPixel);
    	
        //The amount of samples shaded per pixel. Compile time constant. Used to scale contributions back down.
        constexpr auto numShadedSamples = ReSTIRSettings::numReservoirsPerPixel * (1 + (ReSTIRSettings::enableTemporal ? 1 : 0) + (ReSTIRSettings::enableSpatial ? 1 : 0));

        pixelPointers[0] = a_PreviousPixelData[temporalIndex];
        pixelPointers[1] = a_CurrentPixelData[index];

        //Ensure that the depth of both samples is valid, and then combine them at each depth.
        if (!pixelPointers[0].m_SurfaceFlags && !pixelPointers[1].m_SurfaceFlags)
        {
            toCombine[0] = a_PreviousReservoirs[temporalReservoirIndex];
            toCombine[1] = a_CurrentReservoirs[reservoirIndex];

            //Discard samples that are too different.
            const float depth1 = pixelPointers[0].m_IntersectionT;
            const float depth2 = pixelPointers[1].m_IntersectionT;
            const float depthDifPct = fabs(depth1 - depth2) / ((depth1 + depth2) / 2.f);

            const float angleDif = dot(pixelPointers[0].m_Normal, pixelPointers[1].m_Normal);	//Between 0 and 1 (0 to 90 degrees). 
            static constexpr float MAX_ANGLE_COS = 0.72222222223f;	//Dot product is cos of the angle. If higher than this value, it's within 25 degrees.

            //Only do something if the samples are not vastly different.
            if (depthDifPct < 0.10f && angleDif > MAX_ANGLE_COS)
            {
                //Shade before combining the reservoirs into the current reservoir.
                ShadeReservoirs(a_PreviousReservoirs, a_Dimensions.x, temporalPixelX, temporalPixelY, currentPixelX, currentPixelY, a_OutputBuffer);

                //Cap sample count at 20x current to reduce temporal influence. Would grow infinitely large otherwise.
                toCombine[0].sampleCount = min(toCombine[0].sampleCount, toCombine[1].sampleCount * 20);

                if (ReSTIRSettings::enableBiased)
                {
                    CombineBiased(&a_CurrentReservoirs[index], 2, &toCombine[0], &pixelPointers[1], WangHash(a_Seed + index));
                }
                else
                {
                    CombineUnbiased(&a_CurrentReservoirs[index], &pixelPointers[1], 2, &toCombine[0], &pixelPointers[0], WangHash(a_Seed + index));
                }
            }
        }
    }
}

__device__ __inline__ void CombineUnbiased(
    Reservoir* a_OutputReservoir, 
    const WaveFront::SurfaceData* a_OutputSurfaceData, 
    int a_Count, 
    Reservoir* a_Reservoirs,
    const WaveFront::SurfaceData* a_SurfaceDatas, 
    const std::uint32_t a_Seed)
{
    //Ensure enough reservoirs are passed.
    assert(a_Count > 1);

    Reservoir output;
    int sampleCountSum = 0;

    for (int index = 0; index < a_Count; ++index)
    {
        //TODO: possible optimization here: no resampling required for the pixel being merged into if applicable.
        auto* otherReservoir = &a_Reservoirs[index];
        LightSample resampled;
        Resample(&otherReservoir->sample, a_OutputSurfaceData, &resampled);

        assert(otherReservoir->weight >= 0.f);
        assert(otherReservoir->sampleCount >= 0);
        assert(otherReservoir->weightSum >= 0.f);
        assert(otherReservoir->sample.solidAnglePdf >= 0.f);

        const float weight = static_cast<float>(otherReservoir->sampleCount) * otherReservoir->weight * resampled.solidAnglePdf;

        assert(!isnan(weight));
        assert(!isinf(weight));
        assert(weight >= 0.f);
    	
        output.Update(resampled, weight, a_Seed);

        sampleCountSum += otherReservoir->sampleCount;
    }

    output.sampleCount = sampleCountSum;

    //Weigh against other pixels to remove bias from their solid angle by re-sampling.
    int correction = 0;

    for (int index = 0; index < a_Count; ++index)
    {
        auto* otherPixel = &a_SurfaceDatas[index];
        LightSample resampled;
        Resample(&output.sample, otherPixel, &resampled);

        if (resampled.solidAnglePdf > 0)
        {
            correction += a_Reservoirs[index].sampleCount;
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

__device__ __inline__ void CombineBiased(
    Reservoir* a_OutputReservoir, 
    int a_Count, 
    Reservoir* a_Reservoirs,
    const WaveFront::SurfaceData* a_SurfaceData, 
    const std::uint32_t a_Seed)
{
    //Ensure enough reservoirs are passed.
    assert(a_Count > 1);

    //Loop over every depth.
    Reservoir output;
    long long int sampleCountSum = 0;

    //Iterate over the intersection data to combine.
    for (int i = 0; i < a_Count; ++i)
    {
        auto* reservoir = &a_Reservoirs[i];

        assert(reservoir->weight >= 0.f);
        assert(reservoir->sampleCount >= 0);
        assert(reservoir->weightSum >= 0.f);
        assert(reservoir->sample.solidAnglePdf >= 0.f);
        
        LightSample resampled;
        Resample(&(reservoir->sample), a_SurfaceData, &resampled);

        const float weight = static_cast<float>(reservoir->sampleCount) * reservoir->weight * resampled.solidAnglePdf;

        assert(!isnan(weight));
        assert(!isinf(weight));
        assert(weight >= 0.f);
    	
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

__device__ __inline__ void Resample(
    LightSample* a_Input, 
    const WaveFront::SurfaceData* a_PixelData, 
    LightSample* a_Output)
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
    //const auto brdf = MicrofacetBRDF(pixelToLightDir, -a_PixelData->m_IncomingRayDirection, a_PixelData->m_Normal,
    //                                 a_PixelData->m_Color, a_PixelData->m_Metallic, a_PixelData->m_Roughness);

    //The unshadowed contribution (contributed if no obstruction is between the light and surface) takes the BRDF,
    //geometry factor and solid angle into account. Also the light radiance.
    //The only thing missing from this is the scaling with the rest of the scene based on the reservoir PDF.
    //Note: No need to multiply with transport factor because this is depth 0. It is always {1, 1, 1}.
    //const auto unshadowedPathContribution = brdf * solidAngle * cosIn * a_Output->radiance;

    float pdf = 0.f;
    const auto bsdf = EvaluateBSDF(a_PixelData->m_MaterialData, a_PixelData->m_Normal, a_PixelData->m_Tangent, -a_PixelData->m_IncomingRayDirection, pixelToLightDir, pdf);
	
    //If contribution to lobe is 0, just discard. Also goes for NAN which is sometimes sadly present with specular vertices.
    const auto added = pdf + bsdf.x + bsdf.y + bsdf.z;
    if(pdf <= EPSILON || isnan(added) || isinf(added))
    {
        a_Output->unshadowedPathContribution = make_float3(0.f, 0.f, 0.f);
        a_Output->solidAnglePdf = 0;
        return;
    }

    const auto unshadowedPathContribution = (bsdf / pdf) * solidAngle * cosIn * a_Output->radiance;
	
    a_Output->unshadowedPathContribution = unshadowedPathContribution;

    assert(unshadowedPathContribution.x >= 0 && unshadowedPathContribution.y >= 0 && unshadowedPathContribution.z >= 0);
    assert(!isnan(unshadowedPathContribution.x));
    assert(!isnan(unshadowedPathContribution.y));
    assert(!isnan(unshadowedPathContribution.z));
    assert(!isinf(unshadowedPathContribution.x));
    assert(!isinf(unshadowedPathContribution.y));
    assert(!isinf(unshadowedPathContribution.z));

    //For the PDF, I take the unshadowed path contribution as a single float value. Average for now.
    //TODO: Maybe use the human eye for scaling (green weighed more).
    a_Output->solidAnglePdf = (unshadowedPathContribution.x + unshadowedPathContribution.y + unshadowedPathContribution.
        z) / 3.f;
}

//__host__ void GenerateWaveFrontShadowRays(Reservoir* a_Reservoirs, const WaveFront::SurfaceData* a_PixelData, WaveFront::AtomicBuffer<WaveFront::ShadowRayData>* a_AtomicBuffer, unsigned a_NumPixels)
//{
//    //Call in parallel.
//    const int blockSize = CUDA_BLOCK_SIZE;
//    const int numBlocks = (a_NumPixels + blockSize - 1) / blockSize;
//
//    //Separate invocations for each depth to add a stride.
//    for (int depth = 0; depth < ReSTIRSettings::numReservoirsPerPixel; ++depth)
//    {
//        GenerateWaveFrontShadowRaysInternal << <numBlocks, blockSize >> > (a_Reservoirs, a_PixelData, a_AtomicBuffer, a_NumPixels, depth);
//    }
//    cudaDeviceSynchronize();
//    CHECKLASTCUDAERROR;
//}
//
//__global__ void GenerateWaveFrontShadowRaysInternal(Reservoir* a_Reservoirs, const WaveFront::SurfaceData* a_PixelData, WaveFront::AtomicBuffer<WaveFront::ShadowRayData>* a_AtomicBuffer, unsigned a_NumPixels, unsigned a_Depth)
//{
//    int index = blockIdx.x * blockDim.x + threadIdx.x;
//    int stride = blockDim.x * gridDim.x;
//
//    for (int i = index; i < a_NumPixels; i += stride)
//    {
//        const WaveFront::SurfaceData* pixel = &a_PixelData[i];
//
//        //Only generate shadow rays for pixels that hit a surface that is not emissive.
//        if(!(pixel->m_SurfaceFlags))
//        {
//            /*
//             * TODO
//             * Note: This currently divides the expected contribution per reservoir by the amount of reservoirs.
//             * It's essentially like scaling down so that the total adds up to 100% if all shadow rays pass.
//             * This does shoot one shadow ray per reservoir, but I think that's needed for accurate results.
//             * If we are really desperate we could average the reservoir results and then send a single shadow ray.
//             */
//
//            //Get the contribution and scale it down based on the number of reservoirs.
//            Reservoir* reservoir = &a_Reservoirs[RESERVOIR_INDEX(i, a_Depth, ReSTIRSettings::numReservoirsPerPixel)];
//
//            //Only send shadow rays for reservoirs that have a valid sample.
//            if(reservoir->weight > 0)
//            {
//                float3 contribution = (reservoir->sample.unshadowedPathContribution * (reservoir->weight / static_cast<float>(ReSTIRSettings::numReservoirsPerPixel)));
//
//                //Generate a ray for this particular reservoir.
//                float3 toLightDir = reservoir->sample.position - pixel->m_Position;
//                const float l = length(toLightDir);
//                toLightDir /= l;
//
//                //TODO: add stride between these.
//
//                //TODO ensure no shadow acne.
//                //TODO: Pass pixel index to shadow ray data.
//                auto data =
//                    WaveFront::ShadowRayData{
//                        pixel->m_PixelIndex,
//                        pixel->m_Position,
//                        toLightDir, l - 0.005f,
//                        contribution,
//                        WaveFront::LightChannel::DIRECT };
//                a_AtomicBuffer->Add(&data);
//            }
//        }
//    }
//}

__host__ void CombineReservoirBuffers(
    Reservoir* a_Reservoirs1, 
    Reservoir* a_Reservoirs2, 
    const WaveFront::SurfaceData* a_SurfaceData, 
    unsigned a_NumReservoirs, 
    unsigned a_Seed)
{
    const int blockSize = CUDA_BLOCK_SIZE;
    const int numBlocks = (a_NumReservoirs + blockSize - 1) / blockSize;
    CombineReservoirBuffersInternal<< <numBlocks, blockSize >> > (a_Reservoirs1, a_Reservoirs2, a_SurfaceData, a_NumReservoirs, a_Seed);
    cudaDeviceSynchronize();
    CHECKLASTCUDAERROR;

}

__global__ void CombineReservoirBuffersInternal(
    Reservoir* a_Reservoirs1, 
    Reservoir* a_Reservoirs2, 
    const WaveFront::SurfaceData* a_SurfaceData, 
    unsigned a_NumReservoirs, 
    unsigned a_Seed)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
	
    for (int i = index; i < a_NumReservoirs; i += stride)
    {
        const auto pixelIndex = i / ReSTIRSettings::numReservoirsPerPixel;
	    const WaveFront::SurfaceData* surface = &a_SurfaceData[pixelIndex];
 
    	//Only shade surfaces that can be shaded.
        if(!surface->m_SurfaceFlags)
        {
            Reservoir toCombine[2];
            toCombine[0] = a_Reservoirs1[i];
            toCombine[1] = a_Reservoirs2[i];

        	//This function only calls combineBiased because at this point there's no information about the second reservoirs original surface.
        	//Assume it is the same surface.
        	//This is fine when combining neighbour samples, as those have already been re-weighed for the target pixel when unbiased mode is enabled.
        	//Outputs to the input buffer.
            CombineBiased(&a_Reservoirs1[i], 2, &toCombine[0], surface, WangHash(a_Seed + i));
        }
    }
}

//This gets the SM ID on the GPU. So Streaming Multiprocessor.
//Allows us to reason about the L1 cache used by a block.
uint32_t __mysmid()
{
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

