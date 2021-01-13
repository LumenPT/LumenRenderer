#pragma once

struct PTServiceLocator
{
    class OptiXRenderer* m_Renderer; // To be replaced with whatever our actual renderer class will be
    class ShaderBindingTableGenerator* m_SBTGenerator;
};