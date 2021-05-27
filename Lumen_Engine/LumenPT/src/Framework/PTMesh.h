#pragma once

#include "Renderer/ILumenResources.h"

#include "Optix/optix_types.h"

#include <map>

struct PTServiceLocator;
class AccelerationStructure;
class PTPrimitive;

// Extension of the base mesh class to account for implementation details relating to APIs used for the path tracing
class PTMesh : public Lumen::ILumenMesh
{
public:
    // Construction requires all the primitives in the mesh, as well as a reference to the services.
    // This is necessary for the construction of the acceleration structures
    PTMesh(std::vector<std::shared_ptr<Lumen::ILumenPrimitive>>& a_Primitives, PTServiceLocator& a_ServiceLocator);

    PTServiceLocator& m_Services;

};