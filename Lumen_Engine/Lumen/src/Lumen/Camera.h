#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/rotate_vector.hpp>

class LumenCamera
{
public:
	LumenCamera();
	LumenCamera(glm::vec3 a_position, glm::vec3 a_up, float yaw, float pitch);
	~LumenCamera();

	void GetVectorData(glm::vec3& eye, glm::vec3& U, glm::vec3& V, glm::vec3& W);

	glm::vec3 m_position = glm::vec3(0.f, 0.f, 0.f);
	glm::vec3 m_worldUp = glm::vec3(0.f, 1.f, 0.f);

	glm::vec3 m_forward = glm::vec3(0.f, 0.f, 1.0f);
	glm::vec3 m_right = glm::vec3(-1.f, 0.f, 0.f);
	glm::vec3 m_up = glm::vec3(0.f, 1.f, 0.f);

	float m_yaw = 0.f;
	float m_pitch = 0.f;

	float m_focalLength = 1.0f;
	float m_aspectRatio = 1.0f;
	float m_fovY = 35.f;

private:
	void UpdateImagePlane();
	void UpdateCameraVectors();

	glm::vec2 m_imagePlaneHalfSize = glm::vec2(1.0f, 1.0f);
};