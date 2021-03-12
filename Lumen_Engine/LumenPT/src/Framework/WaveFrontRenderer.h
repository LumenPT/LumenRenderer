#pragma once
#if defined(WAVEFRONT)
#include "ShaderBindingTableRecord.h"
#include "MemoryBuffer.h"
#include "Camera.h"
#include "PTServiceLocator.h"
#include "../Shaders/CppCommon/WaveFrontDataStructs.h"

#include "Renderer/LumenRenderer.h"

#include <map>
#include <string>
#include <vector>
#include <memory>
#include <array>
#include <Optix/optix_types.h>
#include <CUDA/builtin_types.h>

using GLuint = unsigned;

class OutputBuffer;
class ShaderBindingTableGenerator;

class AccelerationStructure;

namespace Lumen
{
    class ILumenTexture;
    class ILumenPrimitive;
}

class WaveFrontRenderer : public LumenRenderer
{
public:

    struct InitializationData
    {

        uint8_t m_MaxDepth;
        uint8_t m_RaysPerPixel;
        uint8_t m_ShadowRaysPerPixel;

        uint2 m_RenderResolution;
        uint2 m_OutputResolution;

    };

    WaveFrontRenderer(const InitializationData& a_InitializationData);
    ~WaveFrontRenderer();





    // Creates a cuda texture from the provided raw data and sizes. Only works if the pixel format is uchar4.
    std::shared_ptr<Lumen::ILumenTexture> CreateTexture(void* a_PixelData, uint32_t a_Width, uint32_t a_Height) override;

    std::unique_ptr<Lumen::ILumenPrimitive> CreatePrimitive(PrimitiveData& a_PrimitiveData) override;
    std::unique_ptr<MemoryBuffer> InterleaveVertexData(const PrimitiveData& a_MeshData);

    std::shared_ptr<Lumen::ILumenMesh> CreateMesh(std::vector<std::unique_ptr<Lumen::ILumenPrimitive>>& a_Primitives) override;

    std::shared_ptr<Lumen::ILumenMaterial> CreateMaterial(const MaterialData& a_MaterialData) override;

    std::shared_ptr<Lumen::ILumenScene> CreateScene(SceneData a_SceneData) override;
	
    std::shared_ptr<Lumen::ILumenVolume> LumenRenderer::CreateVolume(const std::string& a_FilePath) override;
   

    GLuint TraceFrame();

    Camera m_Camera;

    Lumen::Transform m_TestTransform;

    static inline const uint2 s_minResolution = make_uint2(1, 1);
    static inline const uint8_t s_minDepth = 1;
    static inline const uint8_t s_minRaysPerPixel = 1;
    static inline const uint8_t s_minShadowRaysPerPixel = 1;

private:

    

    enum class PipelineType
    {
        RESOLVE_RAYS,
        RESOLVE_SHADOW_RAYS
    };

    enum class RayBatchTypeIndex
    {
        PRIM_RAYS_PREV_FRAME,
        CURRENT_RAYS,
        SECONDARY_RAYS,
        NUM_RAY_BATCH_TYPES
    };

    enum class HitBufferTypeIndex
    {
        PRIM_HITS_PREV_FRAME,
        CURRENT_HITS,
        NUM_HIT_BUFFER_TYPES
    };

    static constexpr unsigned s_NumRayBatchTypes = static_cast<unsigned>(RayBatchTypeIndex::NUM_RAY_BATCH_TYPES);
    static constexpr unsigned s_NumHitBufferTypes = static_cast<unsigned>(HitBufferTypeIndex::NUM_HIT_BUFFER_TYPES);
    static constexpr float s_MinTraceDistance = 0.1f;
    static constexpr float s_MaxTraceDistance = 5000.f;



    bool Initialize(const InitializationData& a_InitializationData);

    

    void CreateOutputBuffer();

    void CreateDataBuffers();

    void SetupInitialBufferIndices();

    

    

    static void GetRayBatchIndices(
        unsigned a_WaveIndex, 
        const std::array<unsigned, s_NumRayBatchTypes>& a_CurrentIndices, 
        std::array<unsigned, s_NumRayBatchTypes>& a_Indices);

    static void GetHitBufferIndices(
        unsigned a_WaveIndex,
        const std::array<unsigned, s_NumHitBufferTypes>& a_CurrentIndices,
        std::array<unsigned, s_NumHitBufferTypes>& a_Indices);



    PTServiceLocator m_ServiceLocator;

    std::unique_ptr<OutputBuffer> m_OutputBuffer;

    //Data buffers for the wavefront algorithm.

    std::array<unsigned, s_NumRayBatchTypes> m_RayBatchIndices;
    std::array<unsigned, s_NumHitBufferTypes> m_HitBufferIndices;

    //ResultBuffer storing the different PixelBuffers as different light channels;
    std::unique_ptr<MemoryBuffer> m_ResultBuffer;
    //2 PixelBuffers 1 for the different channels in the ResultBuffer and 1 PixelBuffer for the merged results (to allow for up-scaling the output).
    std::unique_ptr<MemoryBuffer> m_PixelBufferMultiChannel;
    std::unique_ptr<MemoryBuffer> m_PixelBufferSingleChannel;
    //2 ray batches, 1 for storing primary rays, other for overwriting secondary rays.
    std::unique_ptr<MemoryBuffer> m_IntersectionRayBatches[s_NumRayBatchTypes];
    //2 intersection buffers, 1 for storing primary intersections, other for overwriting secondary intersections.
    std::unique_ptr<MemoryBuffer> m_IntersectionBuffers[s_NumHitBufferTypes];
    //1 shadow ray batch to overwrite with shadow rays.
    std::unique_ptr<MemoryBuffer> m_ShadowRayBatch;

    //TEMPORARY
    std::unique_ptr<MemoryBuffer> m_LightBufferTemp;

    std::unique_ptr<class Texture> m_Texture;

    uint2 m_RenderResolution;
    uint2 m_OutputResolution;

    uint8_t m_MaxDepth;
    uint8_t m_RaysPerPixel;
    uint8_t m_ShadowRaysPerPixel;

    unsigned int m_FrameCount;

    bool m_Initialized;

};


#endif