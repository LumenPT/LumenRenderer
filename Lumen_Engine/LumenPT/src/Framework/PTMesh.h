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
    PTMesh(std::vector<std::unique_ptr<Lumen::ILumenPrimitive>>& a_Primitives, PTServiceLocator& a_ServiceLocator);

    // Rebuilds the acceleration structure for the mesh from the primitives' acceleration structures
    void UpdateAccelerationStructure();

    // Returns true if the struct is correct, false if it was updated
    bool VerifyStructCorrectness();

    PTServiceLocator& m_Services;

    // The acceleration structure that is made from the primitives of this mesh
    std::unique_ptr<AccelerationStructure> m_AccelerationStructure;
private:
    // Map used to proof check that the instance IDs used within the scene data table match the ones used by the acceleration structure
    std::map<PTPrimitive*, uint32_t> m_LastUsedInstanceIDs;
};