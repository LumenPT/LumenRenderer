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
    WaveFront::WaveFrontRenderer* m_Renderer; // Reference to the wavefront renderer used to path trace the images
    WaveFront::OptixWrapper* m_OptixWrapper; // Reference to the API wrapper which abstracts away the Optix implementation details
#else
    class OptiXRenderer* m_Renderer;
#endif

    class SceneDataTable* m_SceneDataTable; // Reference to the scene data table which contains all material and vertex data for the scenes

};