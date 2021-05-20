#include "PTMesh.h"
#include "AccelerationStructure.h"
#include "PTPrimitive.h"
#include "PTServiceLocator.h"
#include "RendererDefinition.h"
#include "OptixWrapper.h"

#include "glm/mat4x4.hpp"

PTMesh::PTMesh(std::vector<std::shared_ptr<Lumen::ILumenPrimitive>>& a_Primitives, PTServiceLocator& a_ServiceLocator)
    : ILumenMesh(a_Primitives)
    , m_Services(a_ServiceLocator)
{
}
