#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/rotate_vector.hpp>

#include "Lumen.h"

namespace Lumen
{
	class Camera
	{
	public:
		Camera();
		Camera(glm::vec3 a_Position, glm::vec3 a_Up, float a_Yaw, float a_Pitch);
		~Camera();

		//TODO: set lookat
		
		void SetPosition(glm::vec3 a_Position) { m_Position = a_Position; m_DirtyFlag = true; }
		glm::vec3 GetPosition() const { return m_Position; }

		void SetYaw(float a_Yaw) { m_Yaw = a_Yaw; m_DirtyFlag = true; }
		float GetYaw() const { return m_Yaw; }

		void SetPitch(float a_Pitch) { m_Pitch = a_Pitch; m_DirtyFlag = true; }
		float GetPitch() const { return m_Pitch; }
		
		void GetVectorData(glm::vec3& a_Eye, glm::vec3& a_U, glm::vec3& a_V, glm::vec3& a_W);
		void HandleInput();

	private:
		void UpdateImagePlane();
		void UpdateCameraVectors();

		glm::vec3 m_Position = glm::vec3(0.f, 0.f, 0.f);
		glm::vec3 m_WorldUp = glm::vec3(0.f, 1.f, 0.f);

		glm::vec3 m_Forward = glm::vec3(0.f, 0.f, 1.0f);
		glm::vec3 m_Right = glm::vec3(-1.f, 0.f, 0.f);
		glm::vec3 m_Up = glm::vec3(0.f, 1.f, 0.f);

		float m_Yaw = 0.f;
		float m_Pitch = 0.f;

		float m_MovementSpeed = 1.f;
		float m_RotationSpeed = 1.f;

		float m_FocalLength = 1.0f;
		float m_AspectRatio = 1.0f;

		glm::vec2 m_ImagePlaneHalfSize = glm::vec2(1.0f, 1.0f);
		float m_FovY = 35.f;

		bool m_DirtyFlag = true;
	};
}