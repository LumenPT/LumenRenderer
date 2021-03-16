#pragma once

#ifdef  WAVEFRONT
namespace WaveFront
{
    class WaveFrontRenderer2WithAVengeance;
    class OptixWrapper;
}
#endif

struct PTServiceLocator
{

#ifdef  WAVEFRONT
    WaveFront::WaveFrontRenderer2WithAVengeance* m_Renderer;
#else
    class OptiXRenderer* m_Renderer;
#endif

    class SceneDataTable* m_SceneDataTable;

    WaveFront::OptixWrapper* m_OptixWrapper;
};