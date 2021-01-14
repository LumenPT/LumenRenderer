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
	init.m_Resolution = { 800, 600 };
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


	auto& tarTransform = m_LumenPT->m_Scene->m_MeshInstances[0]->m_Transform;

	glm::vec3 pos, scale;
	pos = tarTransform.GetPosition();
	scale = tarTransform.GetScale();
	auto rot = tarTransform.GetRotationEuler();

	ImGui::Begin("ModelBoi 3000");
	ImGui::DragFloat3("Position", &pos[0]);
	ImGui::DragFloat3("Scale", &scale[0]);
	ImGui::DragFloat3("Rotation", &rot[0]);
	ImGui::End();
	tarTransform.SetPosition(pos);
	tarTransform.SetScale(scale);
	tarTransform.SetRotation(rot);
}

void OutputLayer::HandleCameraInput(Camera& a_Camera)
{
	constexpr float movementSpeed = 1.0f / 60.f;
	glm::vec3 movementDirection = glm::vec3(0.f, 0.f, 0.f);
	glm::vec3 eye, U, V, W;
	a_Camera.GetVectorData(eye, U, V, W);
	
	if (Lumen::Input::IsKeyPressed(LMN_KEY_UP) || Lumen::Input::IsKeyPressed(LMN_KEY_W))
	{
		movementDirection += glm::normalize(W) * movementSpeed;
	}
	if (Lumen::Input::IsKeyPressed(LMN_KEY_DOWN) || Lumen::Input::IsKeyPressed(LMN_KEY_S))
	{
		movementDirection -= glm::normalize(W) * movementSpeed;
	}
	if (Lumen::Input::IsKeyPressed(LMN_KEY_LEFT) || Lumen::Input::IsKeyPressed(LMN_KEY_A))
	{
		movementDirection += glm::normalize(U) * movementSpeed;
	}
	if (Lumen::Input::IsKeyPressed(LMN_KEY_RIGHT) || Lumen::Input::IsKeyPressed(LMN_KEY_D))
	{
		movementDirection -= glm::normalize(U) * movementSpeed;
	}
	if (Lumen::Input::IsKeyPressed(LMN_KEY_LEFT_SHIFT) || Lumen::Input::IsKeyPressed(LMN_KEY_LEFT_CONTROL))
	{
		movementDirection -= glm::normalize(V) * movementSpeed;
	}
	if (Lumen::Input::IsKeyPressed(LMN_KEY_SPACE))
	{
		movementDirection += glm::normalize(V) * movementSpeed;
	}

	if(glm::length(movementDirection))
	{
		a_Camera.SetPosition(eye + glm::normalize(movementDirection) * movementSpeed);
	}

	constexpr float rotationSpeed = 30.f * (1.0f / 60.f);
	float yawRotation = 0.f;
	if (Lumen::Input::IsKeyPressed(LMN_KEY_Q))
	{
		yawRotation += rotationSpeed;
	}
	if(Lumen::Input::IsKeyPressed(LMN_KEY_E))
	{
		yawRotation -= rotationSpeed;
	}

	a_Camera.IncrementYaw(glm::radians(yawRotation));
	//a_Camera.SetYaw(a_Camera.GetYaw() + yawRotation);
}
