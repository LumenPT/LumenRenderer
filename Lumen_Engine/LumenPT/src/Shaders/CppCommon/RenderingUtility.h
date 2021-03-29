#pragma once

#include <cuda_runtime.h>
#include <cuda/helpers.h>
#include "../../src/CUDAKernels/RandomUtilities.cuh"
#include <sutil/Matrix.h>
#include <sutil/Quaternion.h>
#include <sutil/vec_math.h>
#include <math.h>
#include <limits>

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
    float a2 = roughness * roughness;
    float NdotH = fmax(dot(N, H), 0.0f);
    float NdotH2 = NdotH * NdotH;

    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0f) + 1.0f);
    denom = M_PIf * denom * denom;

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

    //const float3 ks = F;
    const float3 kd = (1.0f - F) * (1.0f - metallic);
    const float3 diffuse = albedo / M_PIf;

    return kd * diffuse + specular;
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
    /*
     * No GLM, so manually gotta do this :( Copied over from GLM.
     * Construct the quaternion rotation between the two inputs.
     */
    sutil::Quaternion rotation = sutil::Quaternion();
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
            rotation = sutil::Quaternion(M_PIf, rotationAxis);
        }
        //Normal case.
        else
        {
            rotation = sutil::Quaternion(a_V1, a_V2);
        }
    }

    //The returned matrix.
    sutil::Matrix3x3 m;

    const float qw = rotation[0];
    const float qx = rotation[1];
    const float qy = rotation[2];
    const float qz = rotation[3];

    m[0 * 4 + 0] = 1.0f - 2.0f * qy * qy - 2.0f * qz * qz;
    m[0 * 4 + 1] = 2.0f * qx * qy - 2.0f * qz * qw;
    m[0 * 4 + 2] = 2.0f * qx * qz + 2.0f * qy * qw;

    m[1 * 4 + 0] = 2.0f * qx * qy + 2.0f * qz * qw;
    m[1 * 4 + 1] = 1.0f - 2.0f * qx * qx - 2.0f * qz * qz;
    m[1 * 4 + 2] = 2.0f * qy * qz - 2.0f * qx * qw;

    m[2 * 4 + 0] = 2.0f * qx * qz - 2.0f * qy * qw;
    m[2 * 4 + 1] = 2.0f * qy * qz + 2.0f * qx * qw;
    m[2 * 4 + 2] = 1.0f - 2.0f * qx * qx - 2.0f * qy * qy;

    return m;
}


/*
 * Calculate the PDF for a pair of directions on a surface with the given roughness and normal.
 */
__device__ __forceinline__ float PDFWorldSpace(const float3& a_ViewDir, const float3& a_SurfaceNormal, const float3& a_OutDir, const float a_Roughness)
{
    assert(dot(a_ViewDir, a_SurfaceNormal) <= 0.f);

    const auto transform = RotateAlign(a_SurfaceNormal, float3{ 0.f, 0.f, 1.f });

    const float3 localView = transform * -a_ViewDir;
    const float3 localOut = transform * a_OutDir;


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

    assert(fabsf(dot(microFacetNormal, microFacetNormal) - 1.f) < 0.00001);
    assert(fabsf(dot(localView, localView) - 1.f) < 0.00001);
    assert(fabsf(dot(a_OutDir, a_OutDir) - 1.f) < 0.00001);

    return mfNormalPdf / (4.f * dot(localView, microFacetNormal));
}

__device__ __forceinline__ void SampleGGXVNDF(float3 a_ViewDir, float a_Roughness, float U1, float U2, float3& a_OutputDirection, float& a_OutputPdf)
{
    //Ensure that the view direction goes towards the surface. This is all in local space, with Up = (0, 0, 1).
    //The view direction is inverted already so a view that's not larger than 0 will never actually hit the surface.
    assert(a_ViewDir.z > 0.f);

    // Section 3.2: transforming the view direction to the hemisphere configuration
    float3 Vh = normalize(float3{ a_Roughness * a_ViewDir.x, a_Roughness * a_ViewDir.y, a_ViewDir.z });

    // Section 4.1: orthonormal basis (with special case if cross product is zero)
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = lensq > 0.f ? float3{-Vh.y, Vh.x, 0.f} * rsqrtf(lensq) : float3{ 1.f, 0.f, 0.f };
    float3 T2 = cross(Vh, T1);

    // Section 4.2: parameterization of the projected area
    float r = sqrt(U1);
    float phi = 2.f * M_PIf * U2;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5f * (1.0f + Vh.z);
    t2 = (1.f - s) * sqrt(1.f - t1 * t1) + s * t2;

    // Section 4.3: reprojection onto hemisphere
    float3 Nh = t1 * T1 + t2 * T2 + sqrt(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;

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
    a_OutputDirection = reflect(-a_ViewDir, microFacetNormal);

    assert(fabsf(dot(microFacetNormal, microFacetNormal) - 1.f) < 0.00001);
    assert(fabsf(dot(a_ViewDir, a_ViewDir) - 1.f) < 0.00001);
    assert(fabsf(dot(a_OutputDirection, a_OutputDirection) - 1.f) < 0.00001);


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
    assert(fabsf(dot(a_ViewDirection, a_ViewDirection) - 1.f) < 0.00001);

    a_Seed = WangHash(a_Seed);

    //Uniformly distributed random floats between 0 and 1.
    const float r1 = RandomFloat(a_Seed);
    const float r2 = RandomFloat(a_Seed);

    //Invert of the view direction (leaving the surface).
    const auto inverseView = -a_ViewDirection;

    //Convert the view direction from world to local space. In local space, +z is up.
    const auto transform = RotateAlign(a_SurfaceNormal, float3{ 0.f, 0.f, 1.f });
    const float3 localView = transform * inverseView;

    //Generate a random direction on the hemisphere. This is in normal space where z is the surface normal.
    float pdf;
    float3 sample;
    SampleGGXVNDF(localView, a_Roughness, r1, r2, sample, pdf);

    //The sample is in local space (normal = 0,0,1). Convert to world space.
    const auto transformBack = RotateAlign(float3{ 0.f, 0.f, 1.f }, a_SurfaceNormal);
    const float3 sampleWorld = transformBack * sample;

    assert(fabsf(dot(sample, sample) - 1.f) < 0.00001);

    a_OutputPdf = pdf;
    a_OutputDirection = sampleWorld;
}