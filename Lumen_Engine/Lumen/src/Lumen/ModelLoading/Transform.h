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
        DependentBase(void* a_InstancePointer)
            : m_InstancePointer(a_InstancePointer) {}
        virtual void UpdateDependent() = 0;
        // Need a pointer to the instance when removing a dependent from a vector.
        const void* m_InstancePointer;
    };

    template<typename Type>
    class Dependent : public DependentBase
    {
    public:
        Dependent(Type& a_Instance)
            : DependentBase(&a_Instance)
            , m_Instance(a_Instance) {}
        void UpdateDependent() override
        {
            m_Instance.DependencyCallback();
        };

    private:
        Type& m_Instance;
    };


    class Transform
    {

        friend Transform;
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

        glm::mat4 GetWorldTransformationMatrix() const;
        glm::mat4 GetLocalTransformationMatrix() const;

        operator glm::mat4() const;

        Transform& operator*=(const Lumen::Transform& a_Other);

        template<typename Type>
        void AddDependent(Type& a_Dependent)
        {
            m_Dependents.push_back(std::make_unique<Dependent<Type>>(a_Dependent));
        };

        template<typename Type>
        void RemoveDependent(Type& a_Dependent)
        {
            // Find the instance by its address in memory
            void* ptr = &a_Dependent;
            auto iter = std::find_if(m_Dependents.begin(), m_Dependents.end(), [ptr](std::unique_ptr<DependentBase>& a_Member)
                {
                    return a_Member->m_InstancePointer == ptr;
                });
            if (iter != m_Dependents.end())
            {
                // In the case that m_Dependents is a big vector,
                // swapping the target to the back of the vector will make deleting it from the vector faster
                std::iter_swap(iter, m_Dependents.end() - 1);
                m_Dependents.pop_back();
            }
        }

        Transform* GetParent() const { return m_Parent; }
        const std::vector<Transform*>& GetChildren() const { return m_Children; }

        void SetParent(Transform* a_ParentTransform);
        void AddChild(Transform& a_ChildTransform);
        void RemoveChild(Transform& a_ChildTransform);

        static uint64_t m_IdCount;
        uint64_t m_ID;
    private:

        void SetParentInternal(Transform* a_ParentTransform);
        void AddChildInternal(Transform& a_ChildTransform);
        void RemoveChildInternal(Transform& a_ChildTransform);

        void MakeWorldDirty();
        void MakeLocalDirty();
        void UpdateLocalMatrix() const;
        void UpdateWorldMatrix() const;

        void Decompose();

        void UpdateDependents();


        glm::vec3 m_Position;
        glm::quat m_Rotation;
        glm::vec3 m_Scale;

        mutable glm::mat4 m_WorldMatrix;
        mutable bool m_WorldMatrixDirty;
        mutable glm::mat4 m_LocalMatrix;
        mutable bool m_LocalMatrixDirty;

        Transform* m_Parent;
        std::vector<Transform*> m_Children;

        std::vector<std::unique_ptr<DependentBase>> m_Dependents;

    };

    Transform operator*(const Lumen::Transform& a_Left, const Lumen::Transform& a_Right);
}