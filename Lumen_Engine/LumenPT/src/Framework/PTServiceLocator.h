#pragma once

struct PTServiceLocator
{

#ifdef  WAVEFRONT
    class WaveFrontRenderer* m_Renderer;
#else
    class OptiXRenderer* m_Renderer;
#endif

    class ShaderBindingTableGenerator* m_SBTGenerator;

};