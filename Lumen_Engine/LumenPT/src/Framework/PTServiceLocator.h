#pragma once

#ifdef  WAVEFRONT
namespace WaveFront
{
    class WaveFrontRenderer;
    class OptixWrapper;
}
#endif

struct PTServiceLocator
{

#ifdef  WAVEFRONT
    WaveFront::WaveFrontRenderer* m_Renderer;
    WaveFront::OptixWrapper* m_OptixWrapper;
#else
    class OptiXRenderer* m_Renderer;
#endif

    class SceneDataTable* m_SceneDataTable;

};