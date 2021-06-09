#pragma once

#ifdef  WAVEFRONT
namespace WaveFront
{
    class WaveFrontRenderer;
    class OptixWrapper;
    class DX11Wrapper;
}
#endif

struct PTServiceLocator
{

#ifdef  WAVEFRONT
    WaveFront::WaveFrontRenderer* m_Renderer; // Reference to the wavefront renderer used to path trace the images
    WaveFront::OptixWrapper* m_OptixWrapper; // Reference to the API wrapper which abstracts away the Optix implementation details
    WaveFront::DX11Wrapper* m_DX11Wrapper; // Reference to DX11Wrapper
#else
    class OptiXRenderer* m_Renderer;
#endif

    class SceneDataTable* m_SceneDataTable; // Reference to the scene data table which contains all material and vertex data for the scenes

};