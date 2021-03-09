#include "lmnpch.h"
#include "Transform.h"

#include <glm/gtx/matrix_decompose.inl>

Lumen::Transform::Transform()
    : m_Position(0.0f)
    , m_Rotation(1.0f, 0.0f, 0.0f, 0.0f)
    , m_Scale(1.0f)
    , m_MatrixDirty(true)
    , m_TransformationMatrix(1.0f)
{

}

Lumen::Transform::Transform(const glm::mat4& a_TransformationMatrix)
    : m_TransformationMatrix(a_TransformationMatrix)
    , m_MatrixDirty(false)
{
    Decompose();
}

Lumen::Transform::~Transform()
{
}

Lumen::Transform::Transform(const Transform& a_Other)
{
    a_Other.UpdateMatrix();
    m_Position = a_Other.m_Position;
    m_Rotation = a_Other.m_Rotation;
    m_Scale = a_Other.m_Scale;
    m_TransformationMatrix = a_Other.m_TransformationMatrix;
}

Lumen::Transform& Lumen::Transform::operator=(const Transform& a_Other)
{
    a_Other.UpdateMatrix();
    m_Position = a_Other.m_Position;
    m_Rotation = a_Other.m_Rotation;
    m_Scale = a_Other.m_Scale;
    m_TransformationMatrix = a_Other.m_TransformationMatrix;

    return *this;
}

Lumen::Transform& Lumen::Transform::operator=(const glm::mat4& a_TransformationMatrix)
{
    m_TransformationMatrix = a_TransformationMatrix;
    Decompose();
    return *this;
}

void Lumen::Transform::SetPosition(const glm::vec3& a_NewPosition)
{
    m_Position = a_NewPosition;
    MakeDirty();
}

void Lumen::Transform::SetRotation(const glm::quat& a_Quaternion)
{
    m_Rotation = a_Quaternion;
    MakeDirty();
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
    MakeDirty();
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

glm::mat4 Lumen::Transform::GetTransformationMatrix() const
{
    UpdateMatrix();
    return m_TransformationMatrix;
}

Lumen::Transform::operator glm::mat<4, 4, float, glm::defaultp>() const
{
    return GetTransformationMatrix();
}

Lumen::Transform& Lumen::Transform::operator*=(const Lumen::Transform& a_Other)
{
    m_TransformationMatrix *= a_Other.GetTransformationMatrix();
    Decompose();

    return *this;
}

void Lumen::Transform::MakeDirty()
{
    UpdateDependents();
    m_MatrixDirty = true;
}

void Lumen::Transform::UpdateMatrix() const
{
    if (!m_MatrixDirty)
        return;

    m_TransformationMatrix = glm::mat4(1.0f);

    glm::mat4 rotation = glm::mat4_cast(m_Rotation);

    m_TransformationMatrix = glm::translate(m_TransformationMatrix, m_Position);
    m_TransformationMatrix = m_TransformationMatrix * rotation;
    m_TransformationMatrix = glm::scale(m_TransformationMatrix, m_Scale);

    m_MatrixDirty = false;

}

void Lumen::Transform::Decompose()
{
    // Needed for glm::decompose
    glm::vec3 skew;
    glm::vec4 perspective;

    glm::decompose(m_TransformationMatrix, m_Scale, m_Rotation, m_Position, skew, perspective);
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
    auto mat = a_Left.GetTransformationMatrix() * a_Right.GetTransformationMatrix();
    Transform t(mat);

    return t;
}
