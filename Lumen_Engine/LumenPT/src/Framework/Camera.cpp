#include "Camera.h"

#include <assert.h>

#include <glm/gtx/rotate_vector.hpp>

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

void Camera::SetRotation(glm::quat a_Rotation)
{
	glm::vec3 viewDirection = glm::vec3(0.f, 0.f, 1.0f) * a_Rotation;
	SetRotation(viewDirection);
}

void Camera::SetRotation(glm::vec3& direction)
{
	//https://gamedev.stackexchange.com/questions/112565/finding-pitch-yaw-values-from-lookat-vector
	m_Pitch = glm::degrees(glm::asin(direction.y));
	m_Yaw = glm::degrees(glm::atan(direction.x, direction.z));
}

void Camera::SetLookAt(glm::vec3 a_Position, glm::vec3 a_LookAtPos, glm::vec3 a_WorldUp)
{
	glm::vec3 viewDirection = a_LookAtPos - a_Position;
	SetRotation(viewDirection);
	
	m_Position = a_Position;
	m_WorldUp = a_WorldUp;

	m_DirtyFlag = true;
}

void Camera::GetVectorData(glm::vec3& a_Eye, glm::vec3& a_U, glm::vec3& a_V, glm::vec3& a_W)
{
	if(m_DirtyFlag)
	{
		UpdateValues();
	}

	a_Eye = m_Position;

	a_U = m_Right * m_ImagePlaneHalfSize.x;

	a_V = m_Up * m_ImagePlaneHalfSize.y;

	a_W = m_Forward * m_FocalLength;
}

void Camera::UpdateValues()
{
	UpdateImagePlane();
	UpdateCameraVectors();
}

void Camera::UpdateImagePlane()
{
	m_ImagePlaneHalfSize.y = m_FocalLength * glm::tan(glm::radians(m_FovY) * 0.5f);
	m_ImagePlaneHalfSize.x = m_ImagePlaneHalfSize.y * m_AspectRatio;
}

void Camera::UpdateCameraVectors()
{	
	m_Forward = glm::vec3(0, 0, 1);
	m_Forward = glm::rotate(m_Forward, glm::radians(m_Yaw), m_WorldUp);

	m_Right = glm::normalize(glm::cross(m_WorldUp, m_Forward));
	m_Forward = glm::rotate(m_Forward, glm::radians(m_Pitch), m_Right);

	m_Up = glm::normalize(glm::cross(m_Forward, m_Right));

	m_DirtyFlag = false;
}