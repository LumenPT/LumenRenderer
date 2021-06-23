#include "ILumenScene.h"

Lumen::ILumenScene::Node* Lumen::ILumenScene::Node::AddChild()
{
    m_ChildNodes.push_back(std::make_unique<Node>());
    m_Transform.AddChild(m_ChildNodes.back()->m_Transform);
    m_ChildNodes.back()->m_Parent = this;
    return m_ChildNodes.back().get();
}

void Lumen::ILumenScene::Node::AddChild(std::unique_ptr<Lumen::ILumenScene::Node>& a_Node)
{
    if (a_Node->m_Parent)
        a_Node->m_Parent->RemoveChild(a_Node);

    //m_ChildNodes.pu
}

void Lumen::ILumenScene::Node::RemoveChild(std::unique_ptr<Lumen::ILumenScene::Node>& a_Node)
{
    /*auto fIter = std::find(m_ChildNodes.begin(), m_ChildNodes.end(), &a_Node);
    if (fIter != m_ChildNodes.end())
    {
        m_ChildNodes.erase(fIter);
    }
    m_Transform.RemoveChild(a_Node->m_Transform);

    a_Node->m_Parent = nullptr;
    a_Node->m_Transform.SetParent(nullptr);*/
}

Lumen::ILumenScene::Node* Lumen::ILumenScene::Node::GetFirstIntermediateNode(const Node* a_ParentNode) const
{
    auto p = m_Parent;
    Node* n = nullptr;
    while (p != nullptr)
    {
        if (p == a_ParentNode)
        {
            return n;
        }
        n = p;
        p = p->m_Parent;
    }
}

bool Lumen::ILumenScene::Node::IsChildOf(const Node& a_Node) const
{
    auto p = m_Parent;
    while (p != nullptr)
    {
        if (p == &a_Node)
        {
            return true;
        }
        p = p->m_Parent;
    }
    return false;
}

Lumen::MeshInstance* Lumen::ILumenScene::AddMesh()
{
    m_MeshInstances.push_back(std::make_unique<MeshInstance>());
    return m_MeshInstances.back().get();
}

Lumen::VolumeInstance* Lumen::ILumenScene::AddVolume()
{
    m_VolumeInstances.push_back(std::make_unique<VolumeInstance>());
    return m_VolumeInstances.back().get();
}

void Lumen::ILumenScene::Clear()
{
    m_VolumeInstances.clear();
    m_MeshInstances.clear();
}
