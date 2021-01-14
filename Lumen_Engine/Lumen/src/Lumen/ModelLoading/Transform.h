#pragma once

#include "glm/mat4x4.hpp"
#include "glm/vec3.hpp"
#include "glm/gtx/quaternion.hpp"

#include <vector>
#include <memory>

namespace Lumen // BYOUTIFUL KARTOSHKA
{
    class DependentBase
    {
    public:
        virtual void UpdateDependent() = 0;
    };

    template<typename Type>
    class Dependent : public DependentBase
    {
    public:
        Dependent(Type& a_Instance)
            : m_Instance(a_Instance){};
        void UpdateDependent() override
        {
            m_Instance.DependencyCallback();
        };

    private:
        Type& m_Instance;
    };


    class Transform
    {
    public:
        Transform();
        ~Transform();

        Transform(const Transform& a_Other);
        Transform& operator=(const Transform& a_Other);

        // Constructor and assignment operator from glm::mat4
        Transform(const glm::mat4& a_TransformationMatrix);
        Transform& operator=(const glm::mat4& a_TransformationMatrix);

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

        template<typename Type>
        void AddDependent(Type& a_Dependent)
        {
            m_Dependents.push_back(std::make_unique<Dependent<Type>>(a_Dependent));
        };

    private:
        void MakeDirty();
        void UpdateMatrix() const;

        void Decompose();

        void UpdateDependents();

        glm::vec3 m_Position;
        glm::quat m_Rotation;
        glm::vec3 m_Scale;

        mutable glm::mat4 m_TransformationMatrix;
        mutable bool m_MatrixDirty;

        std::vector<std::unique_ptr<DependentBase>> m_Dependents;
    };

    Transform operator*(const Lumen::Transform& a_Left, const Lumen::Transform& a_Right);
}




