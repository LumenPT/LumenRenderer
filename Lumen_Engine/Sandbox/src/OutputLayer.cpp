#include "OutputLayer.h"

//#include "LumenPT.h"
//#include "Framework/Camera.h"

#ifdef WAVEFRONT
#include "../../LumenPT/src/Framework/WaveFrontRenderer.h"
#else
#include "../../LumenPT/src/Framework/OptiXRenderer.h"
#endif

#include "Lumen/Input.h"
#include "Lumen/ModelLoading/SceneManager.h"
#include "Lumen/KeyCodes.h"

#include "Glad/glad.h"

#include "imgui/imgui.h"

#include "GLFW/glfw3.h"

#include "filesystem"
#include <iostream>

OutputLayer::OutputLayer()
	: m_CameraMovementSpeed(300.0f)
	, m_CameraMouseSensitivity(0.2f)
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

	LumenRenderer::InitializationData init{};
	init.m_RenderResolution = { 800, 600 };
	init.m_OutputResolution = { 800, 600 };
	init.m_MaxDepth = 1;
	init.m_RaysPerPixel = 1;
	init.m_ShadowRaysPerPixel = 1;
//#ifdef WAVEFRONT
//	init.m_RenderResolution = { 800, 600 };
//	init.m_OutputResolution = { 800, 600 };
//	init.m_MaxDepth = 1;
//	init.m_RaysPerPixel = 1;
//	init.m_ShadowRaysPerPixel = 1;
//#else
//#endif

#ifdef WAVEFRONT

	m_Renderer = std::make_unique<WaveFront::WaveFrontRenderer>();

	WaveFront::WaveFrontSettings settings{};
	settings.depth = 3;
	settings.minIntersectionT = 0.1f;
	settings.maxIntersectionT = 5000.f;
	settings.renderResolution = { 800, 600 };
	settings.outputResolution = { 800, 600 };

	static_cast<WaveFront::WaveFrontRenderer*>(m_Renderer.get())->Init(settings);

#else
	m_Renderer = std::make_unique<OptiXRenderer>(init);
#endif
}

OutputLayer::~OutputLayer()
{										    
	glDeleteProgram(m_Program);
}

void OutputLayer::OnUpdate(){

	HandleCameraInput(*m_Renderer->m_Scene->m_Camera);
	auto texture = m_Renderer->TraceFrame(m_Renderer->m_Scene); // TRACE SUM
	HandleSceneInput();

	glBindTexture(GL_TEXTURE_2D, texture);
	glUseProgram(m_Program);
	glDrawArrays(GL_TRIANGLES, 0, 3);
}

void OutputLayer::OnImGuiRender()
{
	ImGuiCameraSettings();
    if (!m_Renderer->m_Scene->m_MeshInstances.empty())
    {
		auto& tarTransform = m_Renderer->m_Scene->m_MeshInstances[0]->m_Transform;

		glm::vec3 pos, scale, rot;
		pos = tarTransform.GetPosition();
		scale = tarTransform.GetScale();
		rot = tarTransform.GetRotationEuler();

		ImGui::Begin("ModelBoi 3000");
		ImGui::DragFloat3("Position", &pos[0]);
		ImGui::DragFloat3("Scale", &scale[0]);
		ImGui::DragFloat3("Rotation", &rot[0]);
		tarTransform.SetPosition(pos);
		tarTransform.SetScale(scale);

		auto deltaRotation = rot - tarTransform.GetRotationEuler();

		glm::quat deltaQuat = glm::quat(glm::radians(deltaRotation));
		if (ImGui::Button("Reset Rotation"))
		{
			tarTransform.SetRotation(glm::vec3(0.0f));
		}
		else
		{
			tarTransform.Rotate(deltaQuat);
		}
		ImGui::End();

    }

	if (!m_Renderer->m_Scene->m_VolumeInstances.empty())
	{
		auto& tarTransform = m_Renderer->m_Scene->m_VolumeInstances[0]->m_Transform;

		glm::vec3 pos, scale, rot;
		pos = tarTransform.GetPosition();
		scale = tarTransform.GetScale();
		rot = tarTransform.GetRotationEuler();

		ImGui::Begin("VolumeBoi 5000");
		ImGui::DragFloat3("Position", &pos[0]);
		ImGui::DragFloat3("Scale", &scale[0]);
		ImGui::DragFloat3("Rotation", &rot[0]);
		
		tarTransform.SetPosition(pos);
		tarTransform.SetScale(scale);

		auto deltaRotation = rot - tarTransform.GetRotationEuler();

		glm::quat deltaQuat = glm::quat(glm::radians(deltaRotation));

		if (ImGui::Button("Reset Rotation"))
		{
			tarTransform.SetRotation(glm::vec3(0.0f));
		}
		else
		{
		    tarTransform.Rotate(deltaQuat);		    
		}
		ImGui::End();
	}
}

void OutputLayer::InitializeScenePresets()
{
	ScenePreset pres = {};


	//temporary stuff to avoid absolute paths to gltf file
	std::filesystem::path p = std::filesystem::current_path();
	std::string assetPath{ p.string() };
	std::replace(assetPath.begin(), assetPath.end(), '\\', '/');

	assetPath += "/Sandbox/assets/models/";

	auto& scene = m_Renderer->m_Scene;
	auto assetManager = m_LayerServices->m_SceneManager;

	// Sample scene loading preset
	pres.m_Key = LMN_KEY_1;
	pres.m_Function = [this, assetManager, assetPath]()
	{
	    auto fileName = "Lantern.gltf";

		auto scene = m_Renderer->m_Scene;
		// I suggest doing a scene clear so that we don't end up with scenes with duplicate meshes
		scene->Clear();	

		auto res = assetManager->LoadGLTF(fileName, assetPath);

		auto mesh = scene->AddMesh();
		auto meshLight = scene->AddMesh();
		mesh->SetMesh(res->m_MeshPool[0]); // We just take the first mesh for simplicity of having a model
		meshLight->SetMesh(res->m_MeshPool[0]);
	};
	m_ScenePresets.push_back(pres);

	pres.m_Key = LMN_KEY_2;
	pres.m_Function = [this, assetManager, assetPath]()
	{
		auto fileName = "Sponza.gltf";
		auto scene = m_Renderer->m_Scene;
		scene->Clear();

		auto res = assetManager->LoadGLTF(fileName, assetPath);

		auto mesh = scene->AddMesh();
		mesh->SetMesh(res->m_MeshPool[0]); // The file has a single mesh, which is sponza itself
		mesh->m_Transform.SetScale(glm::vec3(0.1f));
	};
	m_ScenePresets.push_back(pres);
}


void OutputLayer::HandleCameraInput(Camera& a_Camera)
{
	if (!ImGui::IsAnyItemActive() && Lumen::Input::IsMouseButtonPressed(GLFW_MOUSE_BUTTON_LEFT))
	{
		auto delta = Lumen::Input::GetMouseDelta();

		a_Camera.IncrementYaw(-glm::radians(delta.first * m_CameraMouseSensitivity));
		a_Camera.IncrementPitch(glm::radians(delta.second * m_CameraMouseSensitivity));

	}


	float movementSpeed = m_CameraMovementSpeed / 60.f;
	
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

	//a_Camera.IncrementYaw(glm::radians(yawRotation));
	//a_Camera.IncrementPitch(glm::radians(pitchRotation));
	//a_Camera.SetYaw(a_Camera.GetYaw() + yawRotation);
}

void OutputLayer::HandleSceneInput()
{
	static uint16_t keyDown = 0;

	if (!ImGui::IsAnyItemActive())
	{
		if (!keyDown)
		{
			for (auto preset : m_ScenePresets)
			{
				if (Lumen::Input::IsKeyPressed(preset.m_Key))
				{
					preset.m_Function();
					keyDown = preset.m_Key;

					break;
				}
			}
		}
		else
		{
			if (!Lumen::Input::IsKeyPressed(keyDown))
				keyDown = 0;
		}
	}
}

void OutputLayer::ImGuiCameraSettings()
{
	ImGui::Begin("Camera settings");

	auto del = Lumen::Input::GetMouseDelta();

	ImGui::PushItemWidth(80.0f);
	ImGui::DragFloat("Camera Sensitivity", &m_CameraMouseSensitivity, 0.01f, 0.0f, 1.0f, "%.2f");

	ImGui::DragFloat("Camera Movement Speed", &m_CameraMovementSpeed, 0.1f, 0.0f);

	ImGui::End();
}
