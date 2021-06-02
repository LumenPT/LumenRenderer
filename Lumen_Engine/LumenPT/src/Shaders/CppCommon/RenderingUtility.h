#pragma once

#include <cuda_runtime.h>
#include <cuda/helpers.h>
#include "../../src/CUDAKernels/RandomUtilities.cuh"
#include <sutil/Matrix.h>
#include <sutil/Quaternion.h>
#include <sutil/vec_math.h>
#include <math.h>
#include <limits>
#include <cassert>

#ifndef __CUDA_ARCH__
#include "glm/gtx/quaternion.hpp"
#endif

static constexpr float EPSILON = std::numeric_limits<float>::epsilon();

__device__ __forceinline__ void orthgraphicProjection(float3& origin, float3& direction, int2 launchIndices, int2 screenResolution, const float3 eye, const float3 U, const float3 V, const float3 W)
{
    origin = make_float3(static_cast<float>(launchIndices.x) / screenResolution.x, static_cast<float>(launchIndices.y) / screenResolution.y, 0.0f);
    origin.x = -(origin.x * 2.0f - 1.0f); //we inverse the result, because U image coordinate points left while X vector points right
    origin.y = -(origin.y * 2.0f - 1.0f); //we inverse the result, because V image coordinate points down while Y vector points up
    origin = origin.x * U + origin.y * V;
    origin += eye;
	
    direction = normalize(W);
}

__device__ __forceinline__ void perspectiveProjection(float3& origin, float3& direction, int2 launchIndices, int2 screenResolution, const float3 eye, const float3 U, const float3 V, const float3 W)
{	
    direction = make_float3(static_cast<float>(launchIndices.x) / screenResolution.x, static_cast<float>(launchIndices.y) / screenResolution.y, 0.f);
    direction.x = -(direction.x * 2.0f - 1.0f); //we inverse the result, because U image coordinate points left while X vector points right
    direction.y = -(direction.y * 2.0f - 1.0f); //we inverse the result, because V image coordinate points down while Y vector points up
    //direction = normalize(direction.x * U + direction.y * V + W);
    direction = normalize(direction.x * U + direction.y * V + W);
	
    origin = eye;
}

__device__ __forceinline__ float3 schlick(const float cosTheta, const float3 F0)
{
    return F0 + (1.f - F0) * powf(1.0f - cosTheta, 5.0f);
}

__device__ __forceinline__ float geometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;

    float num = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return num / denom;
}

__device__ __forceinline__ float geometrySmith(const float3 N, const float3 V, const float3 L, float roughness)
{
    float NdotV = fmax(dot(N, V), 0.0f);
    float NdotL = fmax(dot(N, L), 0.0f);
    float ggx2 = geometrySchlickGGX(NdotV, roughness);
    float ggx1 = geometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

__device__ __forceinline__ float distributionGGX(const float3 N, const float3 H, float roughness)
{
    float a = roughness * roughness;
    float a2 = a*a;
    float NdotH = fmax(dot(N, H), 0.0f);
    float NdotH2 = NdotH * NdotH;

    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0f) + 1.0f);
    denom = fmaxf(0.00001f, M_PIf * denom * denom);

    assert(denom != 0);

    return nom / denom;
}

static __forceinline__ __device__ float3 reflectionVector(const float3 win, const float3 n)
{
    return win - 2 * dot(win, n) * n;
}

__device__ __forceinline__ float3 MicrofacetBRDF(const float3 WIN, const float3 WOUT, const float3 N, const float3 albedo, const float metallic, const float roughness)
{
    //const float3 BRDF;

    const float3 H = normalize(WIN + WOUT);

    float3 F0 = make_float3(0.04f);
    F0 = lerp(F0, albedo, metallic);

    const float3 F = schlick(fmax(dot(H, WOUT), 0.f), F0);
    float G = geometrySmith(N, WIN, WOUT, roughness);
    float NDF = distributionGGX(N, H, roughness);

    const float3 numerator = NDF * G * F;
    float denominator = 4.0 * fmax(dot(N, WIN), 0.0f) * fmax(dot(N, WOUT), 0.0f);
    const float3 specular = numerator / fmax(denominator, 0.001f);

    const float3 kd = (1.0f - F) * (1.0f - metallic);
    const float3 diffuse = albedo / M_PIf;

    const auto brdf = kd * diffuse + specular;

    assert(brdf.x >= 0.f && brdf.y >= 0.f && brdf.z >= 0.f);
    assert(!isnan(brdf.x));
    assert(!isnan(brdf.y));
    assert(!isnan(brdf.z));
    assert(!isinf(brdf.x));
    assert(!isinf(brdf.y));
    assert(!isinf(brdf.z));

    return brdf;
}

/*
 * BRDF PDF Distribution functions to sample on the hemispere randomly.
 */






__device__ __forceinline__ float G1(const float3& a_ViewDirSquared, float a_RoughnessSquared)
{
    const float f = (-1 + sqrtf(1 + ((a_RoughnessSquared * a_ViewDirSquared.x + a_RoughnessSquared * a_ViewDirSquared.y) / (a_ViewDirSquared.z)))) / 2.f;
    return 1.f / (1.f + f);
}

__device__ __forceinline__ float D(const float3& a_DirectionSquared, float a_RoughnessSquared)
{
    const float f = ((a_DirectionSquared.x / a_RoughnessSquared) + (a_DirectionSquared.y / a_RoughnessSquared) + a_DirectionSquared.z);
    return 1.f / (M_PIf * a_RoughnessSquared * (f * f));
}

__device__ __forceinline__ sutil::Matrix3x3 RotateAlign(float3 a_V1, float3 a_V2)
{
    assert(fabsf(length(a_V1) - 1.f) < EPSILON * 4.f);
    assert(fabsf(length(a_V2) - 1.f) < EPSILON * 4.f);

    /*
     * No GLM, so manually gotta do this :( Copied over from GLM.
     * Construct the quaternion rotation between the two inputs.
     *
     * NOTE:
     * Optix Quaternions don't work at all. I Don't use them ever and definitely don't use any of their functionality.
     */
    float4 rotation{ 0.f, 0.f, 0.f, 0.f };    //Quaternion.
    float cosTheta = dot(a_V1, a_V2);
    float3 rotationAxis;

    //Not pointing in the same direction
    if (cosTheta < 1.f - EPSILON) 
    {
        //Opposites, exception.
        if (cosTheta < -1.f + EPSILON)
        {
            // special case when vectors in opposite directions :
            // there is no "ideal" rotation axis
            // So guess one; any will do as long as it's perpendicular to start
            // This implementation favors a rotation around the Up axis (Y),
            // since it's often what you want to do.
            rotationAxis = cross(float3{ 0.f, 0.f, 1.f }, a_V1);

            if (dot(rotationAxis, rotationAxis) < EPSILON) // bad luck, they were parallel, try again!
                rotationAxis = cross(float3{ 1.f, 0.f, 0.f }, a_V1);

            rotationAxis = normalize(rotationAxis);

            const float halfPi = M_PIf * 0.5;
            rotationAxis *= sinf(halfPi);
            rotation.x = (cosf(halfPi));
            rotation.y = (rotationAxis.x);
            rotation.z = (rotationAxis.y);
            rotation.w = (rotationAxis.z);
        }
        //Normal case.
        else
        {
            rotationAxis = cross(a_V1, a_V2);

            float s = sqrtf((1.f + cosTheta) * 2.f);
            float invs = 1.f / s;
            rotation.x = (0.5f * s);
            rotation.y = (rotationAxis.x * invs);
            rotation.z = (rotationAxis.y * invs);
            rotation.w = (rotationAxis.z * invs);
        }
    }

    sutil::Matrix3x3 rotationMatrix;

    float rotationxx(rotation.y * rotation.y);
    float rotationyy(rotation.z * rotation.z);
    float rotationzz(rotation.w * rotation.w);
    float rotationxz(rotation.y * rotation.w);
    float rotationxy(rotation.y * rotation.z);
    float rotationyz(rotation.z * rotation.w);
    float rotationwx(rotation.x * rotation.y);
    float rotationwy(rotation.x * rotation.z);
    float rotationwz(rotation.x * rotation.w);

    rotationMatrix[0 * 3 + 0] = 1.f - 2.f * (rotationyy + rotationzz);
    rotationMatrix[1 * 3 + 0] = 2.f * (rotationxy + rotationwz);
    rotationMatrix[2 * 3 + 0] = 2.f * (rotationxz - rotationwy);

    rotationMatrix[0 * 3 + 1] = 2.f * (rotationxy - rotationwz);
    rotationMatrix[1 * 3 + 1] = 1.f - 2.f * (rotationxx + rotationzz);
    rotationMatrix[2 * 3 + 1] = 2.f * (rotationyz + rotationwx);

    rotationMatrix[0 * 3 + 2] = 2.f * (rotationxz + rotationwy);
    rotationMatrix[1 * 3 + 2] = 2.f * (rotationyz - rotationwx);
    rotationMatrix[2 * 3 + 2] = 1.f - 2.f * (rotationxx + rotationyy);

    //The returned matrix.
    return rotationMatrix;
}


/*
 * Calculate the PDF for a pair of directions on a surface with the given roughness and normal.
 */
__device__ __forceinline__ float PDFWorldSpace(const float3& a_ViewDir, const float3& a_SurfaceNormal, const float3& a_OutDir, const float a_Roughness)
{
    assert(dot(a_ViewDir, a_SurfaceNormal) <= 0.f);

    const auto transform = RotateAlign(a_SurfaceNormal, float3{ 0.f, 0.f, 1.f });

    const float3 localView = normalize(transform * -a_ViewDir);
    const float3 localOut = normalize(transform * a_OutDir);


    //Calculate the microfacet normal from the in and out direction.
    const float3 microFacetNormal = normalize((localView + localOut) / 2.f);

    const float roughnessSquared = a_Roughness * a_Roughness;
    const float3 viewDirSquared = localView * localView;
    const float3 mfNormalSquared = microFacetNormal * microFacetNormal;

    //Apply the smith GGX distribution functions.
    const auto g = G1(viewDirSquared, roughnessSquared);
    const auto d = D(mfNormalSquared, roughnessSquared);

    //The probability density of generating this particular normal on the surface with the incoming view direction.
    auto mfNormalPdf = (g * fmaxf(0.f, dot(localView, microFacetNormal)) * d) / localView.z; //In the paper, part of it says that this is the dot product of V.Z (Z = up). This is probably a mistake because elsewhere they write out dot() entirely, and in code samples they dividy by v.z.

    assert(fabsf(length(microFacetNormal) - 1.f) < 0.0001);
    assert(fabsf(length(localView) - 1.f) < 0.0001);
    assert(fabsf(length(a_OutDir) - 1.f) < 0.0001);

    return mfNormalPdf / (4.f * dot(localView, microFacetNormal));
}

__device__ __forceinline__ void SampleGGXVNDF(const float3& a_ViewDir, float a_Roughness, float U1, float U2, float3& a_OutputDirection, float& a_OutputPdf)
{
    //Ensure that the view direction goes towards the surface. This is all in local space, with Up = (0, 0, 1).
    //The view direction is inverted already so a view that's not larger than 0 will never actually hit the surface.
    //This will result in a negative PDF.
    assert(a_ViewDir.z > 0.f);
    assert(fabsf(length(a_ViewDir) - 1.f) < 0.01f);

    // Section 3.2: transforming the view direction to the hemisphere configuration
    float3 Vh = normalize(float3{ a_Roughness * a_ViewDir.x, a_Roughness * a_ViewDir.y, a_ViewDir.z });

    // Section 4.1: orthonormal basis (with special case if cross product is zero)
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;

#if  defined(__CUDA_ARCH__)
    float3 T1 = lensq > 0.f ? float3{-Vh.y, Vh.x, 0.f} * rsqrtf(lensq) : float3{ 1.f, 0.f, 0.f };
#else
    float3 T1 = lensq > 0.f ? float3{ -Vh.y, Vh.x, 0.f } * glm::inversesqrt(lensq) : float3{ 1.f, 0.f, 0.f };
#endif


    float3 T2 = cross(Vh, T1);

    // Section 4.2: parameterization of the projected area
    float r = sqrtf(U1);
    float phi = 2.f * M_PIf * U2;
    float t1 = r * cosf(phi);
    float t2 = r * sinf(phi);
    float s = 0.5f * (1.0f + Vh.z);
    t2 = (1.f - s) * sqrtf(1.f - t1 * t1) + s * t2;

    // Section 4.3: reprojection onto hemisphere
    float3 Nh = t1 * T1 + t2 * T2 + sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;

    // Section 3.4: transforming the normal back to the ellipsoid configuration
    //The way this works: Nh.z is not scaled by the roughness. This means that mirroring surfaces will pretty much always generate the surface normal (smooth).
    //Rough surfaces will deviate further from this (to the point where X, Y and Z all equally contribute to the surface normal). 
    const float3 microFacetNormal = normalize(float3{ a_Roughness * Nh.x, a_Roughness * Nh.y, fmaxf(0.f, Nh.z)});

    //Square the data once and reuse it.
    const float roughnessSquared = a_Roughness * a_Roughness;
    const float3 viewDirSquared = a_ViewDir * a_ViewDir;
    const float3 mfNormalSquared = microFacetNormal * microFacetNormal;

    //Apply the smith GGX distribution functions.
    const auto g = G1(viewDirSquared, roughnessSquared);
    const auto d = D(mfNormalSquared, roughnessSquared);

    //The probability density of generating this particular normal on the surface with the incoming view direction.
    const float mfNormalPdf = (g * fmaxf(0.f, dot(a_ViewDir, microFacetNormal)) * d) / a_ViewDir.z; //In the paper, part of it says that this is the dot product of V.Z (Z = up). This is probably a mistake because elsewhere they write out dot() entirely, and in code samples they dividy by v.z.

    /*
     * NOTE:
     * The reflected direction can actually go into the surface depending on roughness and view direction.
     * This is called inter-reflection, and the new hit surface normal will have to be calculated from the new direction.
     * This has to be repeated until the surface is left.
     * Ask Jacco how he deals with this.
     */

     //Reflect the incoming direction over the retrieved normal. This gives the outgoing direction.
    a_OutputDirection = normalize(reflect(-a_ViewDir, microFacetNormal));

    assert(fabsf(length(microFacetNormal) - 1.f) < 0.01f);
    assert(fabsf(length(a_OutputDirection) - 1.f) < 0.01f);
    //assert(dot(a_ViewDir, microFacetNormal) > 0);


    //Use the Jacobian of the reflection to calculate the PDF for this outgoing direction.
    //The paper uses 4 * dot(v,wg). This seems to result in only half the PDF over the hemisphere at roughness = 1 (I guess it reasons about a full sphere?).
    a_OutputPdf = mfNormalPdf / (4.f * dot(a_ViewDir, microFacetNormal));
}

/*
 * Sample a random reflected direction on a hemisphere for a given view direction on a surface with the given normal and roughness.
 * The roughness specified determines the distribution over the hemisphere.
 * The randomly generated direction is stored in a_OutputDirection.
 * The PDF for the chosen direction is stored in a_OutputPdf.
 */
__device__ __forceinline__ void SampleHemisphere(const float3& a_ViewDirection, const float3& a_SurfaceNormal, const float a_Roughness, unsigned int a_Seed, float3& a_OutputDirection, float& a_OutputPdf)
{
    //Make sure the view direction and normal are normalized.
    assert(fabsf(length(a_ViewDirection) - 1.f) < 0.001f);
    assert(fabsf(length(a_SurfaceNormal) - 1.f) < 0.001f);

    //make sure that the view direction is going towards the normal.
    assert(dot(a_ViewDirection, a_SurfaceNormal) < 0.f);

    a_Seed = WangHash(a_Seed);

    //Uniformly distributed random floats between 0 and 1.
    const float r1 = RandomFloat(a_Seed);
    const float r2 = RandomFloat(a_Seed);

    //Invert of the view direction (leaving the surface).
    const auto inverseView = -a_ViewDirection;

    //Convert the view direction from world to local space. In local space, +z is up.
    const auto transformBack = RotateAlign(float3{ 0.f, 0.f, 1.f }, a_SurfaceNormal);
    const auto transform = RotateAlign(a_SurfaceNormal, float3{ 0.f, 0.f, 1.f });
    float3 localView = transform * inverseView;
    localView.z = fmaxf(0.001f, localView.z);
    localView = normalize(localView);

    //Generate a random direction on the hemisphere. This is in normal space where z is the surface normal.
    float pdf;
    float3 sample;
    SampleGGXVNDF(localView, a_Roughness, r1, r2, sample, pdf);

    //The sample is in local space (normal = 0,0,1). Convert to world space.
    const float3 sampleWorld = normalize(transformBack * sample);

    assert(fabsf(length(sample) - 1.f) < 0.001f);

    a_OutputPdf = pdf;
    a_OutputDirection = sampleWorld;
}