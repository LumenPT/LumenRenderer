#include "Camera.h"

Camera::Camera() :
	m_DirtyFlag(true)
{
	UpdateCameraVectors();
	UpdateImagePlane();
}

Camera::Camera(glm::vec3 a_Position, glm::vec3 a_Up, float a_Yaw, float a_Pitch) :
	m_Position(a_Position),
	m_WorldUp(a_Up),
	m_Yaw(a_Yaw),
	m_Pitch(a_Pitch),
	m_DirtyFlag(true)
{
	UpdateCameraVectors();
	UpdateImagePlane();
}

Camera::~Camera()
{

}

void Camera::GetVectorData(glm::vec3& a_Eye, glm::vec3& a_U, glm::vec3& a_V, glm::vec3& a_W)
{
	if(m_DirtyFlag)
	{
		UpdateCameraVectors();
	}

	a_Eye = m_Position;

	a_U = m_Right * m_ImagePlaneHalfSize.x;

	a_V = m_Up * m_ImagePlaneHalfSize.y;

	a_W = m_Forward * m_FocalLength;
}

/*void Camera::HandleInput()
{
	float movementSpeed = 5.f;
	glm::vec3 movementDirection = glm::vec3(0.f, 0.f, 0.f);

	if (Lumen::Input::IsKeyPressed(LMN_KEY_UP))
	{
		movementDirection += glm::normalize(m_Forward) * movementSpeed;
	}
	if (Lumen::Input::IsKeyPressed(LMN_KEY_DOWN))
	{
		movementDirection -= glm::normalize(m_Forward) * movementSpeed;
	}
	if (Lumen::Input::IsKeyPressed(LMN_KEY_LEFT))
	{
		movementDirection -= glm::normalize(m_Right) * movementSpeed;
	}
	if (Lumen::Input::IsKeyPressed(LMN_KEY_DOWN))
	{
		movementDirection += glm::normalize(m_Right) * movementSpeed;
	}

	m_Position += glm::normalize(movementDirection) * movementSpeed;
}*/

void Camera::UpdateImagePlane()
{
	m_ImagePlaneHalfSize.x = m_FocalLength * glm::tan(glm::radians(m_FovY) * 0.5f);
	m_ImagePlaneHalfSize.y = m_ImagePlaneHalfSize.x * m_AspectRatio;
}

void Camera::UpdateCameraVectors()
{	
	m_Forward = glm::vec3(0, 0, 1);
	m_Forward = glm::rotate(m_Forward, glm::radians(m_Yaw), m_WorldUp);

	m_Right = glm::normalize(glm::cross(m_Forward, m_WorldUp));
	m_Forward = glm::rotate(m_Forward, glm::radians(m_Pitch), m_Right);

	m_Up = glm::normalize(glm::cross(m_Right, m_Forward));

	m_DirtyFlag = false;
}