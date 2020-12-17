#include "Camera.h"

Camera::Camera():
	m_dirtyFlag(true)
{
	UpdateCameraVectors();
	UpdateImagePlane();
}

Camera::Camera(glm::vec3 a_position, glm::vec3 a_up, float yaw, float pitch) :
	m_position(a_position),
	m_worldUp(a_up),
	m_yaw(yaw),
	m_pitch(pitch),
	m_dirtyFlag(true)
{
	UpdateCameraVectors();
	UpdateImagePlane();
}

Camera::~Camera()
{

}

void Camera::GetVectorData(glm::vec3& eye, glm::vec3& U, glm::vec3& V, glm::vec3& W)
{
	UpdateCameraVectors();

	eye = m_position;

	U = glm::normalize(m_right) * m_imagePlaneHalfSize.x;

	V = glm::normalize(m_up) * m_imagePlaneHalfSize.y;

	W = glm::normalize(m_forward) * m_focalLength;
}

void Camera::HandleInput()
{
	float movementSpeed = 5.f;
	glm::vec3 movementDirection = glm::vec3(0.f, 0.f, 0.f);
	
	if (Lumen::Input::IsKeyPressed(LMN_KEY_UP))
	{
		movementDirection += glm::normalize(m_forward) * movementSpeed;
	}
	if (Lumen::Input::IsKeyPressed(LMN_KEY_DOWN))
	{
		movementDirection -= glm::normalize(m_forward) * movementSpeed;
	}
	if (Lumen::Input::IsKeyPressed(LMN_KEY_LEFT))
	{
		movementDirection -= glm::normalize(m_right) * movementSpeed;
	}
	if (Lumen::Input::IsKeyPressed(LMN_KEY_DOWN))
	{
		movementDirection += glm::normalize(m_right) * movementSpeed;
	}

	m_position += glm::normalize(movementDirection) * movementSpeed;
}

void Camera::UpdateImagePlane()
{
	m_imagePlaneHalfSize.x = m_focalLength * glm::tan(glm::radians(m_fovY) * 0.5f);
	m_imagePlaneHalfSize.y = m_imagePlaneHalfSize.x * m_aspectRatio;
}

void Camera::UpdateCameraVectors()
{
	m_forward = glm::vec3(0, 0, 1);
	m_forward = glm::rotate(m_forward, glm::radians(m_yaw), m_worldUp);

	m_right = glm::normalize(glm::cross(m_forward, m_worldUp));
	m_forward = glm::rotate(m_forward, glm::radians(m_pitch), m_right);

	m_up = glm::normalize(glm::cross(m_right, m_forward));
}
