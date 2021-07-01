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
#include "Lumen/Events/ApplicationEvent.h"
#include "Lumen/GLTaskSystem.h"

#include "Tools/FrameSnapshot.h"

#include "Tools/ImGuiUtil.h"

#include "Lumen/ToolUI/ModelLoaderWidget.h"
#include "Lumen/ToolUI/SceneGraph.h"
#include "Lumen/ToolUI/Profiler.h"
#include "../vendor/stb/stb_image_write.h"

#include "Glad/glad.h"

#include "imgui/imgui.h"
#include "imgui/implot.h"

#include "GLFW/glfw3.h"

#include <iostream>


OutputLayer::OutputLayer()
	: m_CameraMovementSpeed(300.0f)
	, m_CameraMouseSensitivity(0.2f)
	, m_CurrentSnapShotIndex(-1)
	, m_CurrentImageBuffer(nullptr)
	, m_SnapshotUV1(0.0f)
    , m_SnapshotUV2(1.0f)
    , m_CurrContentView(NONE)
    , m_ContentViewFunc([](glm::vec2){})
{
	InitContentViewNameTable();
	//auto task = GLTaskSystem::AddTask([this]()
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
		};

	//GLTaskSystem::WaitOnTask(task);
	
}

OutputLayer::~OutputLayer()
{										    
	glDeleteProgram(m_Program);
}

void OutputLayer::OnAttach()
{
	InitializeScenePresets();

	if(!m_Renderer)
	{
		printf("Error: No rendering pipeline present in output layer!");
		return;
	}

	m_ModelLoaderWidget = std::make_unique<ModelLoaderWidget>(*m_LayerServices->m_SceneManager, m_Renderer->m_Scene);
	m_SceneGraph = std::make_unique<Lumen::SceneGraph>();
	m_SceneGraph->SetRendererRef(*m_Renderer);
	m_Profiler = std::make_unique<Profiler>();
}

void OutputLayer::OnUpdate()
{

	if(!m_Renderer)
	{
		printf("Error: No rendering pipeline present in the output layer!");
		return;
	}

	m_Renderer->PerformDeferredOperations();
	HandleCameraInput(*m_Renderer->m_Scene->m_Camera);

	bool recordingSnapshot = false;

	if (Lumen::Input::IsKeyPressed(LMN_KEY_K))
	{
		recordingSnapshot = true;
		m_Renderer->BeginSnapshot();
	}

	if (Lumen::Input::IsKeyPressed(LMN_KEY_EQUAL))
	{
		MakeScreenshot(DefaultScreenshotName());
	}

	auto texture = m_Renderer->GetOutputTexture(); // TRACE SUM
	m_LastFrameTex = texture;
	m_SmallViewportFrameTex = texture;

	auto snap = std::move(m_Renderer->EndSnapshot());
	if (snap)
	{
	    m_FrameSnapshots.push_back(std::move(snap));
	}
	if (texture)
	{
		//GLTaskSystem::AddTask([&]()
		{
			auto err1 = glGetError();

			glBindTexture(GL_TEXTURE_2D, texture);
			glUseProgram(m_Program);
			GLint loc = glGetUniformLocation(m_Program, "a_Gamma");
			glUniform1f(loc, m_Gamma);
			glDrawArrays(GL_TRIANGLES, 0, 3);

			glBindTexture(GL_TEXTURE_2D, 0);
		};
	}
}

void OutputLayer::OnImGuiRender()
{

	if(!m_Renderer)
	{
		printf("Error: No rendering pipeline present in the output layer!");
		return;
	}

	auto tooltip = [](std::string a_Tip)
	{
		if (ImGui::IsItemHovered())
		{
			ImGui::BeginTooltip();
			ImGui::Text(a_Tip.c_str());
			ImGui::EndTooltip();
		}
	};

	if (ImGui::BeginMainMenuBar())
	{
		
	    if (ImGui::BeginMenu("Tools and Debugging"))
	    {
			if (ImGui::Selectable("Scene Graph", m_EnabledTools.m_SceneGraph))
				m_EnabledTools.m_SceneGraph = !m_EnabledTools.m_SceneGraph;

			if (ImGui::Selectable("File Loader", m_EnabledTools.m_FileLoader))
				m_EnabledTools.m_FileLoader = !m_EnabledTools.m_FileLoader;

			if (ImGui::Selectable("Camera Settings", m_EnabledTools.m_CameraSettings))
				m_EnabledTools.m_CameraSettings = !m_EnabledTools.m_CameraSettings;

			if (ImGui::Selectable("Output Image Size settings", m_EnabledTools.m_ImageResizer))
				m_EnabledTools.m_ImageResizer = !m_EnabledTools.m_ImageResizer;

			if (ImGui::Selectable("Pixel Debugger", m_EnabledTools.m_PixelDebugger))
				m_EnabledTools.m_PixelDebugger = !m_EnabledTools.m_PixelDebugger;
			tooltip("Enable a tool that can be used to debug snapshotted pixel buffers");

			if (ImGui::Selectable("Debug Viewport", m_EnabledTools.m_DebugViewport))
				m_EnabledTools.m_DebugViewport = !m_EnabledTools.m_DebugViewport;
			tooltip("Enable a viewport used for debugging off-screen resources");

			if (ImGui::Selectable("Profiler", m_EnabledTools.m_Profiler))
				m_EnabledTools.m_Profiler = !m_EnabledTools.m_Profiler;

			if (ImGui::Selectable("General Settings", m_EnabledTools.m_GeneralSettings))
				m_EnabledTools.m_GeneralSettings = !m_EnabledTools.m_GeneralSettings;

			ImGui::EndMenu();
	    }

		ImGui::Separator();

		if (ImGui::Button("Screenshot"))
			MakeScreenshot(DefaultScreenshotName());

        if (ImGui::IsItemHovered())
        {
			ImGui::BeginTooltip();
			ImGui::Text("Take a screenshot of the current renderer output. File can be found in the Screenshots folder.");
			ImGui::EndTooltip();
        }

		ImGui::EndMainMenuBar();
	}

    if (m_EnabledTools.m_GeneralSettings)
    {
		ImGui::Begin("General Settings");
		auto wvfr = dynamic_cast<WaveFront::WaveFrontRenderer*>(m_Renderer.get());
		if (wvfr)
		{


			auto& denoiserSettings = wvfr->m_DenoiserSettings;

			auto s = "None";
			if (denoiserSettings.m_UseNRD)
				s = "NRD denoising";
			else if (denoiserSettings.m_UseOptix)
				s = "Optix denoising";

			if (ImGui::BeginMenu(s))
			{
				if (ImGui::MenuItem("Optix", 0, denoiserSettings.m_UseOptix))
				{
					denoiserSettings.m_UseOptix = true;
					denoiserSettings.m_UseNRD = false;
				}

                if (ImGui::MenuItem("NRD", 0, denoiserSettings.m_UseNRD))
                {
					denoiserSettings.m_UseOptix = false;
					denoiserSettings.m_UseNRD = true;
                }

				if (ImGui::MenuItem("None", 0, !(denoiserSettings.m_UseOptix || denoiserSettings.m_UseNRD)))
				{
					denoiserSettings.m_UseOptix = false;
					denoiserSettings.m_UseNRD = false;
				}

				ImGui::EndMenu();
			}

            if (denoiserSettings.m_UseOptix)
            {
				ImGui::Checkbox("Albedo", &denoiserSettings.m_OptixAlbedo);
				ImGui::Checkbox("Normal", &denoiserSettings.m_OptixNormal);
				ImGui::Checkbox("Temporal Data",  &denoiserSettings.m_OptixTemporal);
            }
		}
		ImGui::End();
    }

    if (m_EnabledTools.m_CameraSettings)
    {
	    ImGuiCameraSettings();        
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

    // ##ToolBookmark
	// Example of creating a window with an image from a CudaGLTexture object
	if (m_EnabledTools.m_DebugViewport)
	{
		ImGui::Begin("Debug viewport");

		ImGuiUtil::DisplayImage(m_Renderer->m_DebugTexture, glm::ivec2(600, 450));

		ImGui::End();
    }

	// Beginning of the pixel debugger tool, assuming any snapshots have been taken

    if (m_EnabledTools.m_PixelDebugger)
    {
		ImGuiPixelDebugger();
    }

    if (m_EnabledTools.m_ImageResizer)
    {
		auto newRes = m_Renderer->GetRenderResolution();
		ImGui::Begin("Image size settings");
		if (ImGui::BeginMenu("Preset Sizes"))
		{
            for (auto& preset : ms_PresetSizes)
            {
                if (ImGui::Selectable(preset.first.c_str()))
                {
					newRes = preset.second;
                }

				tooltip(std::to_string(preset.second.x) + "x" + std::to_string(preset.second.y));
            }
			ImGui::EndMenu();
		}


		ImGui::DragInt2("Output image dimensions", reinterpret_cast<int*>(&newRes[0]), 0.25f, 0);

		if (newRes != m_Renderer->GetRenderResolution())
		{
			newRes = glm::min(newRes, 4000u);
			m_Renderer->SetRenderResolution(newRes);
		}

		if (ImGui::Button("Screenshot"))
		{
			MakeScreenshot(DefaultScreenshotName());
		}
		ImGui::End();

    }

	auto lastFrame = m_Renderer->GetLastFrameStats();
	m_Profiler->AddUniqueStats(lastFrame);
    if (m_EnabledTools.m_Profiler)
    {
    	m_Profiler->Display();
    }


	/////////////////////////////////////////////////
	// Model loading shenanigans
	/////////////////////////////////////////////////

    if (m_EnabledTools.m_FileLoader)
    {
	    m_ModelLoaderWidget->Display();        
    }

    if (m_EnabledTools.m_SceneGraph)
    {
	    m_SceneGraph->Display(*m_Renderer->m_Scene);        
    }
}

void OutputLayer::OnEvent(Lumen::Event& a_Event)
{
	if (a_Event.GetEventType() == Lumen::EventType::WindowResize)
	{
		auto resizeEvent = static_cast<Lumen::WindowResizeEvent&>(a_Event);
		glViewport(0, 0, resizeEvent.GetWidth(), resizeEvent.GetHeight());
	}
}

void OutputLayer::SetPipeline(const std::shared_ptr<LumenRenderer>& a_Renderer)
{

	m_Renderer = a_Renderer;

}

const std::shared_ptr<LumenRenderer>& OutputLayer::GetPipeline() const
{

	return m_Renderer;

}



void OutputLayer::InitializeScenePresets()
{

	if (!m_Renderer)
	{
		printf("Error: No rendering pipeline present in the output layer!");
		return;
	}

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

	if (!m_Renderer)
	{
		printf("Error: No rendering pipeline present in the output layer!");
		return;
	}

	if (m_Renderer->GetBlendMode() != m_BlendMode)
	{
		m_Renderer->SetBlendMode(m_BlendMode);
	}

	//Toggle between merging and not merging output.
	static std::chrono::time_point<std::chrono::steady_clock> lastToggle = std::chrono::high_resolution_clock::now();
	if (Lumen::Input::IsKeyPressed(LMN_KEY_P))
	{
		//Don't spam it, just toggle once every 500 millis.
		auto now = std::chrono::high_resolution_clock::now();
		if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastToggle).count() > 500)
		{
			lastToggle = now;
			m_Renderer->SetBlendMode(!m_Renderer->GetBlendMode());
			printf("Output append mode is now %s.\n", (m_Renderer->GetBlendMode() ? "on" : "off"));
			m_BlendMode = m_Renderer->GetBlendMode();
		}
	}

    if (!ImGui::IsAnyItemActive() && Lumen::Input::IsMouseButtonPressed(GLFW_MOUSE_BUTTON_LEFT))
	{
		auto delta = Lumen::Input::GetMouseDelta();
		if (delta.first != 0 || delta.second != 0)
		{
			a_Camera.IncrementYaw(-glm::radians(delta.first * m_CameraMouseSensitivity));
			a_Camera.IncrementPitch(glm::radians(delta.second * m_CameraMouseSensitivity));
			m_Renderer->SetBlendMode(false);
		}
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
		m_Renderer->SetBlendMode(false);
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

	const glm::vec2 currentMinMaxRenderDistance = a_Camera.GetMinMaxRenderDistance();
	if( m_MinMaxRenderDistance[0] != currentMinMaxRenderDistance[0] || 
		m_MinMaxRenderDistance[1] != currentMinMaxRenderDistance[1])
	{
		a_Camera.SetMinMaxRenderDistance(glm::vec2(m_MinMaxRenderDistance[0], m_MinMaxRenderDistance[1]));
	}

	//a_Camera.IncrementYaw(glm::radians(yawRotation));
	//a_Camera.IncrementPitch(glm::radians(pitchRotation));
	//a_Camera.SetYaw(a_Camera.GetYaw() + yawRotation);
}

void OutputLayer::ImGuiCameraSettings()
{

    ImGui::Begin("Camera settings");

	auto del = Lumen::Input::GetMouseDelta();

	ImGui::PushItemWidth(80.0f);
	ImGui::DragFloat("Camera Sensitivity", &m_CameraMouseSensitivity, 0.01f, 0.0f, 1.0f, "%.2f");

	ImGui::DragFloat("Camera Movement Speed", &m_CameraMovementSpeed, 0.1f, 0.0f);

	ImGui::SliderFloat("Gamma strength/Brightness", &m_Gamma, 1.0f, 4.0f);

	ImGui::DragFloat("Min render distance", &m_MinMaxRenderDistance[0], 0.5f, 0.001f, m_MinMaxRenderDistance[1], "%.3f", ImGuiSliderFlags_AlwaysClamp);
	ImGui::DragFloat("Max render distance", &m_MinMaxRenderDistance[1], 0.5f, m_MinMaxRenderDistance[0], 0, "%.3f", ImGuiSliderFlags_AlwaysClamp);

	ImGui::Combo("DLSS setting", &m_Dlss_SelectedMode, "Off\0Max performance\0Balanced\0Max quality\0Ultra performance\0Ultra quality\0");
	m_Renderer->m_DlssMode = m_Dlss_SelectedMode;

	ImGui::Checkbox("Converge output", &m_BlendMode);

	ImGui::End();
}

void OutputLayer::ImGuiPixelDebugger()
{
	ImGui::Begin("Pixel debugger and frame snapshots");
	if (!m_FrameSnapshots.empty())
	{

		// Button to clear all snapshots
		if (ImGui::Button("Clear Snapshots"))
			m_FrameSnapshots.clear();

		// Dropdown for all taken snapshots
		if (ImGui::BeginCombo("Snapshots",
			(m_CurrentSnapShotIndex != -1) ? (std::string("Snapshot ") + std::to_string(m_CurrentSnapShotIndex)).c_str() : "No Snapshot selected"))
		{
			for (size_t i = 0; i < m_FrameSnapshots.size(); i++)
			{
				std::string name = "Snapshot " + std::to_string(i);
				if (ImGui::Selectable(name.c_str()))
				{
					m_CurrentSnapShotIndex = i;
					m_CurrentImageBuffer = nullptr;
				}
			}
			ImGui::EndCombo();
		}

		// If a snapshot has been selected
		if (m_CurrentSnapShotIndex != -1)
		{
			// Button to delete the currently selected snapshot
			if (ImGui::Button("Delete Snapshot"))
			{
				m_FrameSnapshots.erase(m_FrameSnapshots.begin() + m_CurrentSnapShotIndex);
				m_CurrentSnapShotIndex = -1;
				m_CurrentImageBuffer = nullptr;
			}

			// Dropdown to select an image buffer to use the tool on
			auto preview = m_CurrentImageBuffer ? m_CurrentImageBuffer->first : "No image buffer selected";
			if (ImGui::BeginCombo("Snapshot buffers", preview.c_str()))
			{
				for (auto& frameSnapshot : m_FrameSnapshots[m_CurrentSnapShotIndex]->GetImageBuffers())
				{
					if (ImGui::Selectable(frameSnapshot.first.c_str()))
					{
						m_CurrentImageBuffer = &frameSnapshot;
					}
				}
				ImGui::EndCombo();
			}
		}

		// If an Image buffer has been selected
		if (m_CurrentImageBuffer)
		{
			// Calc to make the image scale with the window size while keeping its ratio
			auto windowSize = ImGui::GetWindowSize();
			windowSize.x -= 25.0f; // around -25 makes the image look nicely aligned in the middle of the window
			auto imageSize = m_CurrentImageBuffer->second.m_Memory->GetSize();
			auto imageRatio = static_cast<float>(imageSize.x) / static_cast<float>(imageSize.y);

			imageSize = glm::ivec2(windowSize.x, windowSize.x / imageRatio);

			ImGuiUtil::DisplayImage(*m_CurrentImageBuffer->second.m_Memory, imageSize, m_SnapshotUV1, m_SnapshotUV2);

			// Find if the mouse cursor is hovering the image and what its UV coordinates would be
			auto minRect = ImGui::GetItemRectMin();
			auto maxRect = ImGui::GetItemRectMax();
			auto itemSize = ImGui::GetItemRectSize();
			auto mPos = ImGui::GetMousePos();
			auto mouseUV = glm::vec2((mPos.x - minRect.x) / itemSize.x, (mPos.y - minRect.y) / itemSize.y);

			bool mouseOnImage = mPos.x > minRect.x && mPos.x < maxRect.x&& mPos.y > minRect.y && mPos.y < maxRect.y;

			if (mouseOnImage)
			{
				const float scrollSensitivity = 0.02f;
				const float mouseSensitivity = 0.002f;
				auto mouseWheel = -Lumen::Input::GetMouseWheel().y;

				// Zoom in on the mouse cursor with math.
				// Scales the effect of the scroll based on how close to the corners the mouse is to make it work.
				m_SnapshotUV1 -= scrollSensitivity * mouseWheel * mouseUV;
				m_SnapshotUV2 += scrollSensitivity * mouseWheel * (1.0f - mouseUV);

				m_SnapshotUV1 = glm::clamp(m_SnapshotUV1, 0.0f, 1.0f);
				m_SnapshotUV2 = glm::clamp(m_SnapshotUV2, 0.0f, 1.0f);

				glm::vec2 uvSize = m_SnapshotUV2 - m_SnapshotUV1;
				glm::vec2 mouseDelta = glm::vec2(0.0f);
				if (Lumen::Input::IsMouseButtonPressed(GLFW_MOUSE_BUTTON_LEFT))
				{
					auto d = Lumen::Input::GetMouseDelta();
					mouseDelta = glm::vec2(d.first, d.second);
					// Move the UVs based on mouse movement
					m_SnapshotUV1 += mouseSensitivity * mouseDelta * uvSize.x;
					m_SnapshotUV2 += mouseSensitivity * mouseDelta * uvSize.x;

					// Make sure that the movement of the UVs does not leave the [0.0, 1.0] range while keeping the zoom consistent
					if (m_SnapshotUV1.x < 0.0f)
					{
						m_SnapshotUV1.x = 0.0f;
						m_SnapshotUV2.x = uvSize.x;
					}
					else if (m_SnapshotUV2.x > 1.0f)
					{
						m_SnapshotUV2.x = 1.0f;
						m_SnapshotUV1.x = 1.0f - uvSize.x;
					}
					if (m_SnapshotUV1.y < 0.0f)
					{
						m_SnapshotUV1.y = 0.0f;
						m_SnapshotUV2.y = uvSize.y;
					}
					else if (m_SnapshotUV2.y > 1.0f)
					{
						m_SnapshotUV2.y = 1.0f;
						m_SnapshotUV1.y = 1.0f - uvSize.y;
					}
				}

			}

			// Dropdown to select how the texture data is displayed in the tool
			if (ImGui::BeginCombo("Content View Mode", m_ContentViewNames[m_CurrContentView].c_str()))
			{
				ContentViewDropDown();

				ImGui::EndCombo();
			}

			// Display the mouse cursor location and its contents.
			if (mouseOnImage)
			{
				glm::vec2 imageSize = m_CurrentImageBuffer->second.m_Memory->GetSize();
				auto pxlID = glm::ivec2(imageSize * mouseUV);

				ImGui::Text("PixelID: [%i; %i] | UV: [%0.3f, %0.3f]", pxlID.x, pxlID.y, mouseUV.x, mouseUV.y);

				m_ContentViewFunc(mouseUV);
			}

		}
	}
	else
	{
	    ImGui::Text("To begin using this tool, take a snapshot by pressing the 'K' key.");
	}
    ImGui::End();
}

void OutputLayer::InitContentViewNameTable()
{
	m_ContentViewNames[NONE] = "NONE";
	m_ContentViewNames[BYTE] =	"BYTE";
	m_ContentViewNames[BYTE3] = "BYTE3";
	m_ContentViewNames[BYTE4] = "BYTE4";
	m_ContentViewNames[INT] = "INT";
	m_ContentViewNames[INT2] = "INT2";
	m_ContentViewNames[INT3] = "INT3";
	m_ContentViewNames[INT4] = "INT4";
	m_ContentViewNames[FLOAT] = "FLOAT";
	m_ContentViewNames[FLOAT2] = "FLOAT2";
	m_ContentViewNames[FLOAT3] = "FLOAT3";
	m_ContentViewNames[FLOAT4] = "FLOAT4";
}

void OutputLayer::ContentViewDropDown()
{
	// Unavoidable because of the lambda definitions.
	if (ImGui::Selectable(m_ContentViewNames[BYTE].c_str()))
	{
		m_CurrContentView = BYTE;
		m_ContentViewFunc = [&](glm::vec2 a_MouseUV)
		{
			auto content = m_CurrentImageBuffer->second.m_Memory->GetPixel<uint8_t>(a_MouseUV);
			ImGui::Text("X: %u", static_cast<uint32_t>(content));
		};
	}
	if (ImGui::Selectable(m_ContentViewNames[BYTE4].c_str()))
	{
		m_CurrContentView = BYTE4;
		m_ContentViewFunc = [&](glm::vec2 a_MouseUV)
		{
			auto content = m_CurrentImageBuffer->second.m_Memory->GetPixel<glm::vec<4, uint8_t>>(a_MouseUV);
			ImGui::Text("R: %u G: %u B: %u A: %u", static_cast<uint32_t>(content.x), static_cast<uint32_t>(content.y),
				static_cast<uint32_t>(content.z), static_cast<uint32_t>(content.w));
		};
	}
	else if (ImGui::Selectable(m_ContentViewNames[INT].c_str()))
	{
		m_CurrContentView = INT;
		m_ContentViewFunc = [&](glm::vec2 a_MouseUV)
		{
			auto content = m_CurrentImageBuffer->second.m_Memory->GetPixel<int32_t>(a_MouseUV);
			ImGui::Text("X: %i", (content));
		};
	}
	else if (ImGui::Selectable(m_ContentViewNames[INT2].c_str()))
	{
		m_CurrContentView = INT2;
		m_ContentViewFunc = [&](glm::vec2 a_MouseUV)
		{
			auto content = m_CurrentImageBuffer->second.m_Memory->GetPixel<glm::ivec2>(a_MouseUV);
			ImGui::Text("X: %i Y: %i", content.x, content.y);
		};
	}
	else if (ImGui::Selectable(m_ContentViewNames[INT3].c_str()))
	{
		m_CurrContentView = INT3;
		m_ContentViewFunc = [&](glm::vec2 a_MouseUV)
		{
			auto content = m_CurrentImageBuffer->second.m_Memory->GetPixel<glm::ivec3>(a_MouseUV);
			ImGui::Text("X: %i Y: %i Z: %i", content.x, content.y, content.z);
		};
	}
	else if (ImGui::Selectable(m_ContentViewNames[INT4].c_str()))
	{
		m_CurrContentView = INT4;
		m_ContentViewFunc = [&](glm::vec2 a_MouseUV)
		{
			auto content = m_CurrentImageBuffer->second.m_Memory->GetPixel<glm::ivec4>(a_MouseUV);
			ImGui::Text("X: %i Y: %i Z: %i W: %i", content.x, content.y, content.z, content.w);
		};
	}
	else if (ImGui::Selectable(m_ContentViewNames[FLOAT].c_str()))
	{
		m_CurrContentView = FLOAT;
		m_ContentViewFunc = [&](glm::vec2 a_MouseUV)
		{
			auto content = m_CurrentImageBuffer->second.m_Memory->GetPixel<float>(a_MouseUV);
			ImGui::Text("X: %0.3f", content);
		};
	}
	else if (ImGui::Selectable(m_ContentViewNames[FLOAT2].c_str()))
	{
		m_CurrContentView = FLOAT2;
		m_ContentViewFunc = [&](glm::vec2 a_MouseUV)
		{
			auto content = m_CurrentImageBuffer->second.m_Memory->GetPixel<glm::vec2>(a_MouseUV);
			ImGui::Text("X: %0.3f Y: %0.3f", content.x, content.y);
		};
	}
	else if (ImGui::Selectable(m_ContentViewNames[FLOAT3].c_str()))
	{
		m_CurrContentView = FLOAT3;
		m_ContentViewFunc = [&](glm::vec2 a_MouseUV)
		{
			auto content = m_CurrentImageBuffer->second.m_Memory->GetPixel<glm::vec3>(a_MouseUV);
			ImGui::Text("X: %0.3f Y: %0.3f Z: %0.3f", content.x, content.y, content.z);
		};
	}
	else if (ImGui::Selectable(m_ContentViewNames[FLOAT4].c_str()))
	{
		m_CurrContentView = FLOAT4;
		m_ContentViewFunc = [&](glm::vec2 a_MouseUV)
		{
			auto content = m_CurrentImageBuffer->second.m_Memory->GetPixel<glm::vec4>(a_MouseUV);
			ImGui::Text("X: %0.3f Y: %0.3f Z: %0.3f W: %0.3f", content.x, content.y, content.z, content.w);
		};
	}
}

void OutputLayer::MakeScreenshot(std::string a_ScreenshotFileName)
{
	uint32_t w, h;
	auto pixels = m_Renderer->GetOutputTexturePixels(w, h);
	for (size_t i = 0; i < pixels.size(); i += 4)
	{
		pixels[i + 0] = std::powf(static_cast<float>(pixels[i + 0]) / static_cast<float>(255), 1.0f / m_Gamma) * 255;
		pixels[i + 1] = std::powf(static_cast<float>(pixels[i + 1]) / static_cast<float>(255), 1.0f / m_Gamma) * 255;
		pixels[i + 2] = std::powf(static_cast<float>(pixels[i + 2]) / static_cast<float>(255), 1.0f / m_Gamma) * 255;
	}
	std::filesystem::path p = a_ScreenshotFileName;
	std::filesystem::create_directories(p.parent_path());
	auto err = stbi_write_png(a_ScreenshotFileName.c_str(), w, h, 4, pixels.data(), 0);
	assert(err);
}

std::string OutputLayer::DefaultScreenshotName()
{
	time_t now = time(0);
	tm* time = gmtime(&now);
	std::string name = "Screenshot" + std::to_string(time->tm_mday) + std::to_string(time->tm_mon + 1) + '-'
		+ std::to_string(time->tm_hour + 2) + std::to_string(time->tm_min) + std::to_string(time->tm_sec) + ".png";
	return std::filesystem::current_path().string() + "\\Screenshots\\" + name;
}
