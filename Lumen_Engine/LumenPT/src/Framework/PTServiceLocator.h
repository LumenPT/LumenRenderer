#pragma once

struct PTServiceLocator
{
#ifdef WAVEFRONT
    class WaveFrontRenderer* m_Renderer;
#else
    class OptiXRenderer* m_Renderer; // To be replaced with whatever our actual renderer class will be
#endif


    class ShaderBindingTableGenerator* m_SBTGenerator;
};