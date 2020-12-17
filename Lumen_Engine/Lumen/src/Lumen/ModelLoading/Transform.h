#pragma once

#include "glm/mat4x4.hpp"
#include "glm/vec3.hpp"
#include "glm/gtx/quaternion.hpp"

namespace Lumen // BYOUTIFUL KARTOSHKA
{
    class Transform
    {
    public:
        Transform();
        ~Transform();

        Transform(Transform& a_Other);
        Transform& operator=(Transform& a_Other);

        // Constructor and assignment operator from glm::mat4
        Transform(glm::mat4& a_TransformationMatrix);
        Transform& operator=(glm::mat4& a_TransformationMatrix);

        void SetPosition(const glm::vec3& a_NewPosition);
        void SetRotation(const glm::quat& a_Quaternion);
        void SetRotation(const glm::vec3& a_EulerDegrees);
        void SetRotation(const glm::vec3& a_Pivot, float a_Degrees);
        void SetScale(const glm::vec3& a_Scale);

        void Move(const glm::vec3& a_Movement);
        void Rotate(const glm::quat& a_Quaternion);
        void Rotate(const glm::vec3& a_EulerDegrees);
        void Rotate(const glm::vec3& a_Pivot, float a_Degrees);
        void ScaleUp(const glm::vec3 a_Scale);

        void TransformBy(const Transform& a_Other);
        void CopyTransform(const Transform& a_Other);

        const glm::vec3& GetPosition() const;
        const glm::quat& GetRotationQuat() const;
        glm::vec3 GetRotationEuler() const;
        const glm::vec3& GetScale() const;

        glm::mat4 GetTransformationMatrix() const;

        operator glm::mat4() const;

        Transform& operator*=(const Lumen::Transform& a_Other);

    private:
        void MakeDirty();
        void UpdateMatrix() const;

        void Decompose();

        glm::vec3 m_Position;
        glm::quat m_Rotation;
        glm::vec3 m_Scale;

        mutable glm::mat4 m_TransformationMatrix;
        mutable bool m_MatrixDirty;


    };

    Transform operator*(const Lumen::Transform& a_Left, const Lumen::Transform& a_Right);
}

