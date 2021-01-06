#pragma once

#include <cuda_runtime.h>
#include <cinttypes>


//Some defines to make the functions less scary and more readable
#define GPU __device__ __forceinline__ 
#define CPU __global__ __forceinline__
#define CPU_ONLY __host__ __forceinline__



/*
 * Set up buffers and other resources required by the pipeline.
 * Called once at program start, or when buffers need to be resized.
 */
CPU void Initialize(const std::uint32_t a_ScreenWidth, const std::uint32_t a_ScreenHeight, const std::uint32_t a_Depth);

/*
 * Called once before every frame.
 * Generates primary rays from the camera etc.
 * Rays are stored in the ray batch.
 */
CPU void PreRenderSetup(const DeviceCamera& a_Camera);

/*
 * Call Optix to intersect all rays in the ray batch.
 * Stores intersection data in the intersection data buffer.
 */
CPU_ONLY void IntersectRays();
 
/*
 * Shade the intersection points.
 * This does direct, indirect and specular shading.
 * This fills the shadow ray buffer with potential contributions per pixel.
 */
CPU void Shade();

/*
 * This resolves all shadow rays in parallel, and adds the light contribution
 * of each light channel to the output pixel buffer for each un-occluded ray.
 */
CPU void ResolveShadowRays();

/*
 * Apply de-noising, up scaling and post-processing effects.
 */
CPU void PostProcess();


//The below functions are only called internally from the GPU within the above defined functions.

//Called in setup.
GPU void GenerateRays();
GPU void GenerateMotionVectors();

//Called during shading
GPU void ShadeDirect();
GPU void ShadeSpecular();
GPU void ShadeIndirect();

//Called during post-processing.
GPU void Denoise();
GPU void MergeLightChannels();
GPU void DLSS();
GPU void PostProcessingEffects();


/*
 * Example calling order of functions.
 */
inline void Example()
{
    constexpr int w = 720;
    constexpr int h = 540;
    constexpr int depth = 3;
    DeviceCamera camera;

    Initialize(w, h, depth); //Set up memory buffers.

    while(true)
    {
        PreRenderSetup(camera); //Generate primary rays.

        for(int d = 0; d < depth; ++d)
        {
            IntersectRays(); //Gets intersection points for primary and secondary rays.
            Shade(); //Creates shadow rays with their potential contribution. Creates new secondary rays.
        }

        ResolveShadowRays(); //Resolves all shadow rays generated by the shading.
        PostProcess(); //Denoising, upscaling, motion blur dof etc.
    }
}



CPU void Initialize(const std::uint32_t a_ScreenWidth, const std::uint32_t a_ScreenHeight, const std::uint32_t a_Depth)
{
    //TODO:
    /*
     * - Pixel buffer swap chain (once for current and one for previous frame). Each buffer contains the direct, indirect and specular light contribution per pixel (RGB). Primary ray also stored.
     *
     * - Ray batch for primary rays. Size is w * h * sizeof(PrimaryRay). Primary ray has direction, origin, potential contribution scalar.
     *
     * - Shadow ray batch size w * h * 2 * depth * sizeof(ShadowRay). Shadow ray has direction, origin, max distance, potential contribution, light channel index.
     *
     * - Intersection data size w * h * sizeof(IntersectionData). IntersectionData holds ID of the mesh and triangle for later lookup.
     *
     */
}

CPU void PreRenderSetup(const DeviceCamera& a_Camera)
{
    //TODO
    /*
     * - Use the camera to generate rays. These rays 
     * - Implement motion vectors
     */

    //Generate rays based on the camera.
    GenerateRays();

    //Generate motion vectors from the previous frame.
    GenerateMotionVectors();
}

CPU_ONLY void IntersectRays()
{
    //TODO
    /*
     * - Call Optix from the CPU to intersect all the rays in the ray batch. This fills the intersection data buffer.
     */
}

CPU void Shade()
{
    //TODO
    /*
     * - Implement the below functions.
     * - Access to intersection data, as well as the ray resulting in this shading for chained BRDF scaling.
     */

    ShadeIndirect(); //Generate secondary rays.
    ShadeSpecular(); //Generate shadow rays for specular highlights.
    ShadeDirect();   //Generate shadow rays for direct lights.
}

CPU void ResolveShadowRays()
{
    //TODO
    /*
     * - Parallellize the shadow ray resolving to be efficient but also thread safe. Some shadow rays affect the same output pixel.
     * Note:
     * - Only the first depth contributes to direct light. The other depths all add up in indirect.
     * - Each shadow rays potential contribution must already be scaled down by the entire paths bounce BRDFs at this point.
     */
}


CPU void PostProcess()
{
    //TODO
    /*
     * Not needed now. Can be implemented later.
     * For now just merge the final light contributions to get the final pixel color.
     */
    Denoise();
    MergeLightChannels();
    DLSS();
    PostProcessingEffects();
}