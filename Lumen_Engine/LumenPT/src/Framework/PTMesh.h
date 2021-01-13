#pragma once

#include "Renderer/ILumenResources.h"

#include "Optix/optix_types.h"

class PTServiceLocator;
class AccelerationStructure;

class PTMesh : public Lumen::ILumenMesh
{
public:
    PTMesh(std::vector<std::unique_ptr<Lumen::ILumenPrimitive>>& a_Primitives, PTServiceLocator& a_ServiceLocator);

    PTServiceLocator& m_Services;

    std::unique_ptr<AccelerationStructure> m_AccelerationStructure;
};