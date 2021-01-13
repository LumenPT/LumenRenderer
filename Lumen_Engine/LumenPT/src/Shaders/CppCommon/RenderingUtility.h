#pragma once

#include <cuda_runtime.h>
#include <cuda/helpers.h>
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
