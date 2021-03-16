#pragma once
#ifdef WAVEFRONT
#include <Lumen/Renderer/LumenRenderer.h>
#include "../Shaders/CppCommon/WaveFrontDataStructs/OptixLaunchParams.h"
#include "../Shaders/CppCommon/WaveFrontDataStructs/OptixShaderStructs.h"
#include "ShaderBindingTableRecord.h"

#include <Optix/optix.h>
#include <filesystem>
#include <memory>
#include <vector>
#include <map>

class MemoryBuffer;
class ShaderBindingTableGenerator;
class AccelerationStructure;

namespace WaveFront
{

    struct ComputedStackSizes
    {
        int DirectSizeTrace;
        int DirectSizeState;
        int ContinuationSize;
    };

    class OptixWrapper
    {

    public:

        struct InitializationData
        {
            CUcontext m_CUDAContext;

            struct ProgramData
            {
                std::filesystem::path m_ProgramPath;
                std::string m_ProgramLaunchParamName;
                std::string m_ProgramRayGenFuncName;
                std::string m_ProgramMissFuncName;
                std::string m_ProgramAnyHitFuncName;
                std::string m_ProgramClosestHitFuncName;
                uint8_t m_MaxNumPayloads = 2;
                uint8_t m_MaxNumHitResultAttributes = 2;
            }m_ProgramData;

        };

        OptixWrapper(const InitializationData& a_InitializationData);
        ~OptixWrapper();

        template<typename VertexType, typename IndexType = uint32_t>
        std::unique_ptr<AccelerationStructure> BuildGeometryAccelerationStructure(
            std::vector<VertexType> a_Vertices, size_t a_VertexOffset = 0,
            std::vector<IndexType> a_Indices = std::vector<IndexType>(), size_t a_IndexOffset = 0) const;

        std::unique_ptr<AccelerationStructure> BuildGeometryAccelerationStructure(
            const OptixAccelBuildOptions& a_BuildOptions,
            const OptixBuildInput& a_BuildInput) const;

        std::unique_ptr<AccelerationStructure> BuildInstanceAccelerationStructure(std::vector<OptixInstance> a_Instances) const;



        void UpdateSBT();

        void TraceRays(
            unsigned int a_NumRays,
            const OptixLaunchParameters& a_LaunchParams,
            CUstream a_CUDAStream = nullptr) const;

    private:

        bool Initialize(const InitializationData& a_InitializationData);

        bool InitializeContext(CUcontext a_CUDAContext);

        bool CreatePipeline(const InitializationData::ProgramData& a_ProgramData);

        OptixPipelineCompileOptions CreatePipelineOptions(
            const std::string& a_LaunchParamName,
            unsigned int a_NumPayloadValues,
            unsigned int a_NumAttributes) const;

        bool CreatePipeline(const OptixModule& a_Module,
            const OptixPipelineCompileOptions& a_PipelineOptions,
            const std::string& a_RayGenFuncName,
            const std::string& a_MissFuncName,
            const std::string& a_AnyHitFuncName,
            const std::string& a_ClosestHitFuncName,
            OptixPipeline& a_Pipeline);

        OptixModule CreateModule(const std::filesystem::path& a_PtxPath, const OptixPipelineCompileOptions& a_PipelineOptions) const;

        OptixProgramGroup CreateProgramGroup(OptixProgramGroupDesc a_ProgramGroupDesc, const std::string& a_Name);

        void DestroyProgramGroups();

        ProgramGroupHeader GetProgramGroupHeader(const std::string& a_GroupName) const;

        static void AccumulateStackSizes(OptixProgramGroup a_ProgramGroup, OptixStackSizes& a_StackSizes);

        static ComputedStackSizes ComputeStackSizes(OptixStackSizes a_StackSizes, int a_TraceDepth, int a_DirectDepth, int a_ContinuationDepth);

        void SetupPipelineBuffer();

        void SetupShaderBindingTable();

        static void OptixDebugCallback(unsigned int a_Level, const char* a_Tag, const char* a_Message, void*);



        CUcontext m_CudaContex;

        OptixDeviceContext m_DeviceContext;

        OptixPipeline m_Pipeline;

        OptixModule m_Module;

        std::unique_ptr<ShaderBindingTableGenerator> m_SBTGenerator;

        RecordHandle<void> m_RayGenRecord;
        RecordHandle<void> m_HitRecord;
        RecordHandle<void> m_MissRecord;

        std::unique_ptr<MemoryBuffer> m_SBTRecordBuffer;

        std::unique_ptr<MemoryBuffer> m_OptixLaunchParamBuffer;

        std::map<std::string, OptixProgramGroup> m_ProgramGroups;

        bool m_Initialized;

        static constexpr char s_RayGenPGName[] = "RayGenPG";
        static constexpr char s_HitPGName[] = "HitPG";
        static constexpr char s_MissPGName[] = "MissPG";

    };

}

#include "OptixWrapper.inl"

#endif