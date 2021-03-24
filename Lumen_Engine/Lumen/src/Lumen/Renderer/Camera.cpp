#include "Camera.h"

#include <assert.h>

#include <glm/gtx/rotate_vector.hpp>

Camera::Camera() :
	m_DirtyFlag(true)
{
	UpdateCameraVectors();
	UpdateImagePlane();

	m_PreviousFrameMatrix = m_CurrentFrameMatrix;
}

Camera::Camera(glm::vec3 a_Position, glm::vec3 a_Up) :
	m_Position(a_Position),
	m_WorldUp(a_Up),
	m_DirtyFlag(true)
{
	UpdateCameraVectors();
	UpdateImagePlane();

	m_PreviousFrameMatrix = m_CurrentFrameMatrix;
}

Camera::~Camera()
{

}

void Camera::SetPosition(glm::vec3 a_Position)
{
	m_Position = a_Position;
	m_DirtyFlag = true;
}

void Camera::SetRotation(glm::quat a_Rotation)
{
	//glm::vec3 viewDirection = glm::vec3(0.f, 0.f, 1.0f) * a_Rotation;
	//SetRotation(viewDirection);

	m_Rotation = a_Rotation;
}

void Camera::SetLookAt(glm::vec3 a_Position, glm::vec3 a_LookAtPos, glm::vec3 a_WorldUp)
{
	const glm::vec3 viewDirection = a_LookAtPos - a_Position;

	m_Rotation = glm::quatLookAtRH(viewDirection, a_WorldUp);
	m_Position = a_Position;
	m_WorldUp = a_WorldUp;

	m_DirtyFlag = true;
}

void Camera::IncrementYaw(const float& a_AngleInRadians)
{
	if (m_DirtyFlag)
	{
		UpdateValues();
	}
	
	m_Rotation = glm::angleAxis(a_AngleInRadians, glm::vec3(m_WorldUp)) * m_Rotation;
	m_DirtyFlag = true;
}

void Camera::IncrementPitch(const float& a_AngleInRadians)
{
	if (m_DirtyFlag)
	{
		UpdateValues();
	}
	
	m_Rotation = glm::angleAxis(a_AngleInRadians, glm::vec3(m_Right)) * m_Rotation;
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

void Camera::GetMatrixData(glm::mat4& a_PreviousFrameMatrix, glm::mat4& a_CurrentFrameMatrix)
{
	if (m_DirtyFlag)
	{
		UpdateValues();
	}
	
	a_PreviousFrameMatrix = m_PreviousFrameMatrix;
	a_CurrentFrameMatrix = m_CurrentFrameMatrix;
}

glm::mat4 Camera::GetProjectionMatrix() const
{
	return glm::perspective(glm::radians(m_FovY), m_AspectRatio, 0.5f, 10000.f);
}

void Camera::UpdatePreviousFrameMatrix()
{
	m_PreviousFrameMatrix = m_CurrentFrameMatrix;
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
	
	
	m_CurrentFrameMatrix = glm::toMat4(m_Rotation);
	m_CurrentFrameMatrix[3] = glm::vec4(m_Position, 1.0f);

	m_Right = m_CurrentFrameMatrix[0];
	m_Up = m_CurrentFrameMatrix[1];
	m_Forward = m_CurrentFrameMatrix[2];
	
	m_DirtyFlag = false;
}