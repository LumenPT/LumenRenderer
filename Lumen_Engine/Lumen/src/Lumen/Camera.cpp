#include "Camera.h"

LumenCamera::LumenCamera()
{
	m_position = glm::vec3(250.f, 250.f, -900.f);
	m_worldUp = glm::vec3(0.f, 1.f, 0.f);

	UpdateCameraVectors();
	UpdateImagePlane();
}

LumenCamera::LumenCamera(glm::vec3 a_position, glm::vec3 a_up, float yaw, float pitch) :
	m_position(a_position),
	m_worldUp(a_up),
	m_yaw(yaw),
	m_pitch(pitch)
{
	UpdateCameraVectors();
	UpdateImagePlane();
}

LumenCamera::~LumenCamera()
{

}

void LumenCamera::GetVectorData(glm::vec3& eye, glm::vec3& U, glm::vec3& V, glm::vec3& W)
{
	UpdateCameraVectors();

	eye = m_position;

	U = glm::normalize(m_right) * m_imagePlaneHalfSize.x;

	V = glm::normalize(m_up) * m_imagePlaneHalfSize.y;

	W = glm::normalize(m_forward) * m_focalLength;
}

void LumenCamera::UpdateImagePlane()
{
	m_imagePlaneHalfSize.x = m_focalLength * glm::tan(glm::radians(m_fovY) * 0.5f);
	m_imagePlaneHalfSize.y = m_imagePlaneHalfSize.x * m_aspectRatio;
}

void LumenCamera::UpdateCameraVectors()
{
	m_forward = glm::vec3(0, 0, 1);
	m_forward = glm::rotate(m_forward, glm::radians(m_yaw), m_worldUp);

	m_right = glm::normalize(glm::cross(m_forward, m_worldUp));
	m_forward = glm::rotate(m_forward, glm::radians(m_pitch), m_right);

	m_up = glm::normalize(glm::cross(m_right, m_forward));
}
