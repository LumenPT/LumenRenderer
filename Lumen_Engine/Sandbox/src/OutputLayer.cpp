#include "OutputLayer.h"

#include "LumenPT.h"
#include "Framework/Camera.h"
#include "Lumen/Input.h"

#include "Glad/glad.h"

#include "imgui/imgui.h"

#include <iostream>

#include "Lumen/KeyCodes.h"

OutputLayer::OutputLayer()
{
	auto vs = glCreateShader(GL_VERTEX_SHADER);
	auto fs = glCreateShader(GL_FRAGMENT_SHADER);

	glShaderSource(vs, 1, &m_VSSource, nullptr);
	glCompileShader(vs);

	int success;
	char infoLog[512];
	glGetShaderiv(vs, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(vs, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
	};

	glShaderSource(fs, 1, &m_FSSource, nullptr);
	glCompileShader(fs);
	glGetShaderiv(fs, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(fs, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
	};

	auto program = glCreateProgram();

	glAttachShader(program, vs);
	glAttachShader(program, fs);

	glLinkProgram(program);

	glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success)
    {
		glGetProgramInfoLog(program, 512, nullptr, infoLog);
		std::cout << "ERROR::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }

	glDeleteShader(vs);
	glDeleteShader(fs);

	m_Program = program;

	LumenPT::InitializationData init{};

#ifdef WAVEFRONT
	init.m_RenderResolution = { 800, 600 };
	init.m_OutputResolution = { 800, 600 };
	init.m_MaxDepth = 1;
	init.m_RaysPerPixel = 1;
	init.m_ShadowRaysPerPixel = 1;
#else
#endif

	m_LumenPT = std::make_unique<LumenPT>(init);



}

OutputLayer::~OutputLayer()
{
	glDeleteProgram(m_Program);
}

void OutputLayer::OnUpdate(){

	HandleCameraInput(m_LumenPT->m_Camera);
	auto texture = m_LumenPT->TraceFrame(); // TRACE SUM


	glBindTexture(GL_TEXTURE_2D, texture);
	glUseProgram(m_Program);
	glDrawArrays(GL_TRIANGLES, 0, 3);
}

void OutputLayer::OnImGuiRender()
{
    if (!m_LumenPT->m_Scene->m_MeshInstances.empty())
    {
		auto& tarTransform = m_LumenPT->m_Scene->m_MeshInstances[0]->m_Transform;

		glm::vec3 pos, scale, rot;
		pos = tarTransform.GetPosition();
		scale = tarTransform.GetScale();
		rot = tarTransform.GetRotationEuler();

		ImGui::Begin("ModelBoi 3000");
		ImGui::DragFloat3("Position", &pos[0]);
		ImGui::DragFloat3("Scale", &scale[0]);
		ImGui::DragFloat3("Rotation", &rot[0]);

		ImGui::End();
		tarTransform.SetPosition(pos);
		tarTransform.SetScale(scale);
		tarTransform.SetRotation(rot);

    }

	if (!m_LumenPT->m_Scene->m_VolumeInstances.empty())
	{
		auto& tarTransform = m_LumenPT->m_Scene->m_VolumeInstances[0]->m_Transform;

		glm::vec3 pos, scale, rot;
		pos = tarTransform.GetPosition();
		scale = tarTransform.GetScale();
		rot = tarTransform.GetRotationEuler();

		ImGui::Begin("VolumeBoi 5000");
		ImGui::DragFloat3("Position", &pos[0]);
		ImGui::DragFloat3("Scale", &scale[0]);
		ImGui::DragFloat3("Rotation", &rot[0]);

		ImGui::End();
		tarTransform.SetPosition(pos);
		tarTransform.SetScale(scale);
		tarTransform.SetRotation(rot);

	}
}

void OutputLayer::HandleCameraInput(Camera& a_Camera)
{
	float movementSpeed = 1.0f / 60.f;
	if(Lumen::Input::IsKeyPressed(LMN_KEY_SPACE))
	{
		movementSpeed *= 150.f;
	}
	
	glm::vec3 movementDirection = glm::vec3(0.f, 0.f, 0.f);
	glm::vec3 eye, U, V, W;
	a_Camera.GetVectorData(eye, U, V, W);
	
	if (Lumen::Input::IsKeyPressed(LMN_KEY_W))
	{
		movementDirection += glm::normalize(W) * movementSpeed;
	}
	if (Lumen::Input::IsKeyPressed(LMN_KEY_S))
	{
		movementDirection -= glm::normalize(W) * movementSpeed;
	}
	if (Lumen::Input::IsKeyPressed(LMN_KEY_A))
	{
		movementDirection += glm::normalize(U) * movementSpeed;
	}
	if (Lumen::Input::IsKeyPressed(LMN_KEY_D))
	{
		movementDirection -= glm::normalize(U) * movementSpeed;
	}
	if (Lumen::Input::IsKeyPressed(LMN_KEY_Q))
	{
		movementDirection -= glm::normalize(V) * movementSpeed;
	}
	if (Lumen::Input::IsKeyPressed(LMN_KEY_E))
	{
		movementDirection += glm::normalize(V) * movementSpeed;
	}

	if(glm::length(movementDirection))
	{
		a_Camera.SetPosition(eye + glm::normalize(movementDirection) * movementSpeed);
	}

	constexpr float rotationSpeed = 100.f * (1.0f / 60.f);
	float yawRotation = 0.f;
	float pitchRotation = 0.f;
	if (Lumen::Input::IsKeyPressed(LMN_KEY_LEFT))
	{
		yawRotation += rotationSpeed;
	}
	if(Lumen::Input::IsKeyPressed(LMN_KEY_RIGHT))
	{
		yawRotation -= rotationSpeed;
	}
	if (Lumen::Input::IsKeyPressed(LMN_KEY_UP))
	{
		pitchRotation -= rotationSpeed;
	}
	if (Lumen::Input::IsKeyPressed(LMN_KEY_DOWN))
	{
		pitchRotation += rotationSpeed;
	}

	a_Camera.IncrementYaw(glm::radians(yawRotation));
	a_Camera.IncrementPitch(glm::radians(pitchRotation));
	//a_Camera.SetYaw(a_Camera.GetYaw() + yawRotation);
}
