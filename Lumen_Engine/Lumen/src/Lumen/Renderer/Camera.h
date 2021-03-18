#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>

//#include "Lumen.h"

class Camera
{
public:
	Camera();
	Camera(glm::vec3 a_Position, glm::vec3 a_Up = glm::vec3(0.0f,1.0f,0.0f));
	~Camera();

	//TODO: set lookat
	
	void SetPosition(glm::vec3 a_Position) { m_Position = a_Position; m_DirtyFlag = true; }
	glm::vec3 GetPosition() const { return m_Position; }

	void SetRotation(glm::quat a_Rotation);
	
	void SetLookAt(glm::vec3 a_Position, glm::vec3 a_LookAt, glm::vec3 a_WorldUp);

	void IncrementYaw(const float& a_AngleInRadians);

	void IncrementPitch(const float& a_AngleInRadians);

	void SetAspectRatio(float a_AspectRatio) { m_AspectRatio = a_AspectRatio; m_DirtyFlag = true; }
	float GetAspectRatio() { return m_AspectRatio; }
	
	void GetVectorData(glm::vec3& a_Eye, glm::vec3& a_U, glm::vec3& a_V, glm::vec3& a_W);

private:
	void UpdateValues();
	void UpdateImagePlane();
	void UpdateCameraVectors();

	glm::vec3 m_Position = glm::vec3(0.f, 0.f, 0.f);
	glm::vec3 m_WorldUp = glm::vec3(0.f, 1.f, 0.f);

	glm::vec3 m_Forward = glm::vec3(0.f, 0.f, 1.0f);
	glm::vec3 m_Right = glm::vec3(-1.f, 0.f, 0.f);
	glm::vec3 m_Up = glm::vec3(0.f, 1.f, 0.f);

	glm::quat m_Rotation = glm::quat(1.f, 0.f, 0.f, 0.f);

	float m_FocalLength = 1.0f;
	float m_AspectRatio = 1.0f;

	glm::vec2 m_ImagePlaneHalfSize = glm::vec2(1.0f, 1.0f);
	float m_FovY = 90.f;

	bool m_DirtyFlag = true;
};