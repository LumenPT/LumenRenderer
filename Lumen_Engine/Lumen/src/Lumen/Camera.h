#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/rotate_vector.hpp>

#include "Lumen.h"

class Camera
{
public:
	Camera();
	Camera(glm::vec3 a_position, glm::vec3 a_up, float yaw, float pitch);
	~Camera();

	void GetVectorData(glm::vec3& eye, glm::vec3& U, glm::vec3& V, glm::vec3& W);
	void HandleInput();

private:
	void UpdateImagePlane();
	void UpdateCameraVectors();

	glm::vec3 m_position = glm::vec3(0.f, 0.f, 0.f);
	glm::vec3 m_worldUp = glm::vec3(0.f, 1.f, 0.f);

	glm::vec3 m_forward = glm::vec3(0.f, 0.f, 1.0f);
	glm::vec3 m_right = glm::vec3(-1.f, 0.f, 0.f);
	glm::vec3 m_up = glm::vec3(0.f, 1.f, 0.f);

	float m_yaw = 0.f;
	float m_pitch = 0.f;

	float m_movementSpeed = 50.f;

	float m_focalLength = 1.0f;
	float m_aspectRatio = 1.0f;
	float m_fovY = 35.f;
	
	glm::vec2 m_imagePlaneHalfSize = glm::vec2(1.0f, 1.0f);

	bool m_dirtyFlag = true;
};