#pragma once

#include "Renderer/ILumenResources.h"

#include "Optix/optix_types.h"

#include <map>

struct PTServiceLocator;
class AccelerationStructure;
class PTPrimitive;

class PTMesh : public Lumen::ILumenMesh
{
public:
    PTMesh(std::vector<std::unique_ptr<Lumen::ILumenPrimitive>>& a_Primitives, PTServiceLocator& a_ServiceLocator);

    void UpdateAccelerationStructure();

    // Returns true if the struct is correct, false if it was updated
    bool VerifyStructCorrectness();

    PTServiceLocator& m_Services;

    std::unique_ptr<AccelerationStructure> m_AccelerationStructure;
private:
    std::map<PTPrimitive*, uint32_t> m_LastUsedInstanceIDs;
};