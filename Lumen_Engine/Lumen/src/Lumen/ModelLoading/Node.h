//#include "Transform.h"
#include <string>
#include <vector>
#include <memory>

namespace Lumen
{
	class Transform;
	
	struct Node
	{
		std::string m_Name;						// as specified by gltf file
		int m_MeshID = -1;						// index of the mesh in the scene this node refers to. -1 if no mesh
		int m_NodeID = -1;						// unique index of the node in the scene's node array
		std::unique_ptr<Transform> m_LocalTransform;	// Transforms all child transforms with this.
		std::vector<int> m_ChilIndices;			// Indices of child nodes in scene's node array
	};
}