#include "lmnpch.h"
#include "Transform.h"

#include <glm/gtx/matrix_decompose.inl>

uint64_t Lumen::Transform::m_IdCount = 0;

Lumen::Transform::Transform()
    : m_Position(0.0f)
    , m_Rotation(1.0f, 0.0f, 0.0f, 0.0f)
    , m_Scale(1.0f)
    , m_WorldMatrixDirty(true)
    , m_WorldMatrix(1.0f)
    , m_LocalMatrixDirty(true)
    , m_LocalMatrix(0.0f)
    , m_Parent(nullptr)
    , m_ID(m_IdCount++)
{

}

Lumen::Transform::Transform(const glm::mat4& a_TransformationMatrix)
    : m_LocalMatrix(a_TransformationMatrix)
    , m_LocalMatrixDirty(false)
    , m_WorldMatrix(a_TransformationMatrix)
    , m_WorldMatrixDirty(false)
    , m_Parent(nullptr)
    , m_ID(m_IdCount++)
{

    Decompose();
}

Lumen::Transform::~Transform()
{
    if (m_Parent)
        m_Parent->RemoveChild(*this);
    for (auto& child : m_Children)
    {
        child->SetParentInternal(nullptr);
    }
}

Lumen::Transform::Transform(const Transform& a_Other)
{
    a_Other.UpdateWorldMatrix();
    m_Position = a_Other.m_Position;
    m_Rotation = a_Other.m_Rotation;
    m_Scale = a_Other.m_Scale;
    m_WorldMatrix = a_Other.m_WorldMatrix;
    m_LocalMatrixDirty = a_Other.m_LocalMatrixDirty;
    m_WorldMatrixDirty = a_Other.m_WorldMatrixDirty;
    m_Parent = a_Other.GetParent();
    if (m_Parent)
        m_Parent->AddChildInternal(*this);
}

Lumen::Transform& Lumen::Transform::operator=(const Transform& a_Other)
{
    a_Other.UpdateWorldMatrix();
    m_Position = a_Other.m_Position;
    m_Rotation = a_Other.m_Rotation;
    m_Scale = a_Other.m_Scale;
    m_LocalMatrix = a_Other.m_LocalMatrix;
    m_WorldMatrixDirty = a_Other.m_WorldMatrixDirty;
    m_ID = m_IdCount++;

    m_Parent = a_Other.m_Parent;
    if (m_Parent)
    {
        m_Parent->AddChildInternal(*this);
    }

    return *this;
}

Lumen::Transform& Lumen::Transform::operator=(const glm::mat4& a_TransformationMatrix)
{
    m_ID = m_IdCount++;
    m_Parent = nullptr;
    m_LocalMatrix = a_TransformationMatrix;
    Decompose();
    return *this;
}

void Lumen::Transform::SetPosition(const glm::vec3& a_NewPosition)
{
    m_Position = a_NewPosition;
    MakeLocalDirty();
}

void Lumen::Transform::SetRotation(const glm::quat& a_Quaternion)
{
    m_Rotation = a_Quaternion;
    MakeLocalDirty();
}

void Lumen::Transform::SetRotation(const glm::vec3& a_EulerDegrees)
{
    SetRotation(glm::quat(glm::radians(a_EulerDegrees)));
}

void Lumen::Transform::SetRotation(const glm::vec3& a_Pivot, float a_Degrees)
{
    SetRotation(a_Pivot * a_Degrees);
}

void Lumen::Transform::SetScale(const glm::vec3& a_Scale)
{
    m_Scale = a_Scale;
    MakeLocalDirty();
}

void Lumen::Transform::Move(const glm::vec3& a_Movement)
{
    SetPosition(GetPosition() + a_Movement);
}

void Lumen::Transform::Rotate(const glm::quat& a_Quaternion)
{
    SetRotation(GetRotationQuat() * a_Quaternion);
}

void Lumen::Transform::Rotate(const glm::vec3& a_EulerDegrees)
{
    Rotate(glm::quat(glm::radians(a_EulerDegrees)));
}

void Lumen::Transform::Rotate(const glm::vec3& a_Pivot, float a_Degrees)
{
    Rotate(a_Pivot * a_Degrees);
}

void Lumen::Transform::ScaleUp(const glm::vec3 a_Scale)
{
    SetScale(GetScale() + a_Scale);
}

void Lumen::Transform::TransformBy(const Transform& a_Other)
{
    SetPosition(glm::vec4(GetPosition(), 1.0f) * static_cast<glm::mat4>(a_Other));
    SetRotation(GetRotationQuat() * a_Other.GetRotationQuat());
    SetScale(GetScale() * a_Other.GetScale());
}

void Lumen::Transform::CopyTransform(const Transform& a_Other)
{
    SetPosition(a_Other.GetPosition());
    SetRotation(a_Other.GetRotationQuat());
    SetScale(a_Other.GetScale());
}

const glm::vec3& Lumen::Transform::GetPosition() const
{
    return m_Position;
}

const glm::quat& Lumen::Transform::GetRotationQuat() const
{
    return m_Rotation;
}

glm::vec3 Lumen::Transform::GetRotationEuler() const
{
    return glm::degrees(glm::eulerAngles(m_Rotation));
}

const glm::vec3& Lumen::Transform::GetScale() const
{
    return m_Scale;
}

glm::mat4 Lumen::Transform::GetWorldTransformationMatrix() const
{
    UpdateWorldMatrix();
    return m_WorldMatrix;
}

glm::mat4 Lumen::Transform::GetLocalTransformationMatrix() const
{
    UpdateLocalMatrix();
    return m_LocalMatrix;
}

Lumen::Transform::operator glm::mat<4, 4, float, glm::defaultp>() const
{
    return GetWorldTransformationMatrix();
}

Lumen::Transform& Lumen::Transform::operator*=(const Lumen::Transform& a_Other)
{
    m_WorldMatrix *= a_Other.GetWorldTransformationMatrix();
    Decompose();

    return *this;
}

void Lumen::Transform::SetParent(Transform* a_ParentTransform)
{
    SetParentInternal(a_ParentTransform);
    m_Parent->AddChildInternal(*this);
    MakeWorldDirty();
}

void Lumen::Transform::AddChild(Transform& a_ChildTransform)
{
    AddChildInternal(a_ChildTransform);
    a_ChildTransform.SetParentInternal(this);
}

void Lumen::Transform::RemoveChild(Transform& a_ChildTransform)
{
    RemoveChildInternal(a_ChildTransform);
    a_ChildTransform.SetParentInternal(nullptr);

    //TODO: Do we WANT to keep the child in its current world transform?

}

void Lumen::Transform::SetParentInternal(Transform* a_ParentTransform)
{
    auto prevParent = m_Parent;
    m_Parent = a_ParentTransform;
    if (prevParent)
        prevParent->RemoveChildInternal(*this);
    MakeWorldDirty();
}

void Lumen::Transform::AddChildInternal(Transform& a_ChildTransform)
{
    m_Children.push_back(&a_ChildTransform);
}

void Lumen::Transform::RemoveChildInternal(Transform& a_ChildTransform)
{
    auto fIter = std::find(m_Children.begin(), m_Children.end(), &a_ChildTransform);
    if (fIter != m_Children.end())
        m_Children.erase(fIter);
}


void Lumen::Transform::MakeWorldDirty()
{
    UpdateDependents();
    m_WorldMatrixDirty = true;

    for (auto& child : m_Children)
    {
        child->MakeWorldDirty();
    }
}

void Lumen::Transform::MakeLocalDirty()
{
    UpdateDependents();
    m_LocalMatrixDirty = true;
    m_WorldMatrixDirty = true; // If the local transform was changed, then the world transform was also invalidated

    for (auto& child : m_Children)
    {
        child->MakeWorldDirty();
    }
}

void Lumen::Transform::UpdateLocalMatrix() const
{
    if (!m_LocalMatrixDirty)
        return;

    // Build the local transformation matrix
    m_LocalMatrix = glm::mat4(1.0f);

    glm::mat4 rotation = glm::mat4_cast(m_Rotation);

    m_LocalMatrix = glm::translate(m_LocalMatrix, m_Position);
    m_LocalMatrix = m_LocalMatrix * rotation;
    m_LocalMatrix = glm::scale(m_LocalMatrix, m_Scale);

    m_LocalMatrixDirty = false;
}

void Lumen::Transform::UpdateWorldMatrix() const
{
    // If world matrix is still valid, there is nothing necessary to do
    if (!m_WorldMatrixDirty)
        return;

    // Try updating the local matrix
    UpdateLocalMatrix();

    if (m_Parent != nullptr)
    {
        // Get the transformation of the parents transform
        // This will go recursively, rebuilding world transforms until it reaches a parent transform which doesn't need to be rebuilt
        auto parentWorld = m_Parent->GetWorldTransformationMatrix();

        // TODO: Is this the correct order? Run it and find out!
        //m_WorldMatrix = m_LocalMatrix * parentWorld;
        m_WorldMatrix = parentWorld * m_LocalMatrix;
    }
    else
        m_WorldMatrix = m_LocalMatrix;

    // The world matrix was rebuilt, so it's no longer dirty
    // If it was invalidated by the local matrix, that flag was lowered in UpdateLocalMatrix()
    m_WorldMatrixDirty = false;

}

void Lumen::Transform::Decompose()
{
    // Needed for glm::decompose
    glm::vec3 skew;
    glm::vec4 perspective;

    glm::decompose(m_LocalMatrix, m_Scale, m_Rotation, m_Position, skew, perspective);
}

void Lumen::Transform::UpdateDependents()
{
    for (auto& dependent : m_Dependents)
    {
        dependent->UpdateDependent();
    }
}

Lumen::Transform Lumen::operator*(const Lumen::Transform& a_Left, const Lumen::Transform& a_Right)
{
    auto mat = a_Left.GetWorldTransformationMatrix() * a_Right.GetWorldTransformationMatrix();
    Transform t(mat);

    return t;
}