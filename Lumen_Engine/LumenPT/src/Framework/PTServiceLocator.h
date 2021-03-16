#pragma once

struct PTServiceLocator
{

#ifdef  WAVEFRONT
    class WaveFrontRenderer2WithAVengeance* m_Renderer;
#else
    class OptiXRenderer* m_Renderer;
#endif

    class ShaderBindingTableGenerator* m_SBTGenerator;

    class SceneDataTable* m_SceneDataTable;

    class OptixWrapper* m_OptixWrapper;
};