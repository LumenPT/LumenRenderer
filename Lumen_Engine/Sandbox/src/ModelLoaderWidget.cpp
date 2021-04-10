#include "ModelLoaderWidget.h"

#include "ImGui/imgui.h"

#include <map>

namespace fs = std::filesystem;

ModelLoaderWidget::ModelLoaderWidget(Lumen::SceneManager& a_SceneManager, std::shared_ptr<Lumen::ILumenScene>& a_SceneRef)
    : m_SceneManager(a_SceneManager)
    , m_SceneRef(a_SceneRef)
    , m_State(State::Directory)
{
	m_SelectedPath = fs::current_path();
}

void ModelLoaderWidget::Display()
{
	ImGui::SetNextWindowSize(ImVec2(550.0f, 600.0f));
	ImGui::Begin("Model and File loader");

    switch (m_State)
    {
	case State::Directory:
		DirectoryNagivation();
		break;
	case State::Loading:
		LoadModel();
		break;
	case State::ModelLoaded:
		ModelSelection();
		break;
	default:
		abort();
    }


	if (!m_AdditionalMessage.empty())
		ImGui::Text(m_AdditionalMessage.c_str());

	ImGui::End();
}

void ModelLoaderWidget::DirectoryNavigatorHeader()
{
	struct Dir
	{
		std::string m_Name;
		fs::path m_Path;
	};
	std::vector<Dir> dirs;
	auto p = m_SelectedPath;
    do 
    {
		Dir d = { p.filename().string(), p };
		dirs.push_back(d);
		p = p.parent_path();
	} while (p.has_relative_path());

	Dir d = { p.string(), p };
	dirs.push_back(d);

    for (int64_t i = dirs.size() - 1; i >= 0; --i)
    {
		auto& dir = dirs[i];
		if (ImGui::Button(dir.m_Name.c_str()))
			m_SelectedPath = dir.m_Path;

        if (i)
		    ImGui::SameLine();            
    }
}

void ModelLoaderWidget::LoadModel()
{
    if (m_PathToOpen.extension() == ".gltf")
    {
		m_LoadedResource = m_SceneManager.LoadGLTF(m_PathToOpen.filename().string(), m_PathToOpen.parent_path().string() + "\\");
		m_State = State::ModelLoaded;
		m_AdditionalMessage = "";
    }
	else
	{
		m_AdditionalMessage = "Loading of " + m_PathToOpen.extension().string() + " files is not supported yet.";
		m_State = State::Directory;
	}
}

void ModelLoaderWidget::ModelSelection()
{
	ImGui::Text("Model loaded, implementing more shite");

	if (!m_LoadedResource->m_Scenes.empty() && ImGui::ListBoxHeader("Loaded scenes"))
	{
		auto menuRect = ImGui::GetItemRectSize();
		for (size_t i = 0; i < m_LoadedResource->m_Scenes.size(); i++)
		{
			ImGui::Text(m_LoadedResource->m_Scenes[i]->m_Name.c_str());

			auto rMin = ImGui::GetItemRectMin();
			auto rMax = ImGui::GetItemRectMax();

			ImVec2 min = rMin;
			ImVec2 max = ImVec2(rMin.x + menuRect.x, rMax.y + 5);

			if (ImGui::IsMouseHoveringRect(min, max))
			{
				ImGui::SameLine();
				ImGui::SetCursorPosX(menuRect.x - 80.0f);
				ImGui::PushItemWidth(80.0f);
				if (ImGui::Button("Set scene"))
				{
					m_SceneRef = m_LoadedResource->m_Scenes[i];
					m_AdditionalMessage = "Scene successfully set to " + m_LoadedResource->m_Scenes[i]->m_Name;
				}
			}
				//m_SceneRef = m_LoadedResource->m_Scenes[i];
		}
		ImGui::ListBoxFooter();
    }

	TransformSpecifier();

	if (!m_LoadedResource->m_MeshPool.empty() && ImGui::ListBoxHeader("Loaded Meshes"))
	{
		auto menuRect = ImGui::GetItemRectSize();

		for (size_t i = 0; i < m_LoadedResource->m_MeshPool.size(); i++)
		{
			ImGui::Text("Mesh %llu", i);

			auto rMin = ImGui::GetItemRectMin();
			auto rMax = ImGui::GetItemRectMax();

			ImVec2 min = rMin;
			ImVec2 max = ImVec2(rMin.x + menuRect.x, rMax.y + 5);

			if (ImGui::IsMouseHoveringRect(min, max))
			{
				ImGui::SameLine();
				ImGui::SetCursorPosX(menuRect.x - 80.0f);
				ImGui::PushItemWidth(80.0f);
				if (ImGui::Button("Add Mesh"))
				{
					auto m = m_SceneRef->AddMesh();

					m->SetMesh(m_LoadedResource->m_MeshPool[i]);
					m->m_Transform = m_TransformToApply;
                    if (m_ResetTransformOnMeshAdded)
					    m_TransformToApply = Lumen::Transform();

					m_AdditionalMessage = "Mesh successfully added to the scene.";
				}
			}
		}
		ImGui::ListBoxFooter();
	}

	

	if (ImGui::Button("Return to directory view"))
	{
		m_State = State::Directory;
		m_LoadedResource = nullptr;
	}
}

void ModelLoaderWidget::TransformSpecifier()
{
	glm::vec3 t, r, s;
	t = m_TransformToApply.GetPosition();
	r = m_TransformToApply.GetRotationEuler();
	s = m_TransformToApply.GetScale();

	ImGui::Checkbox("Reset transform after adding mesh", &m_ResetTransformOnMeshAdded);
	ImGui::DragFloat3("Position", &t[0]);
	ImGui::DragFloat3("Rotation", &r[0]);
	ImGui::DragFloat3("Scale", &s[0]);
	m_TransformToApply.SetPosition(t);
	m_TransformToApply.SetScale(s);

	auto deltaRotation = r - m_TransformToApply.GetRotationEuler();

	glm::quat deltaQuat = glm::quat(glm::radians(deltaRotation));
	if (ImGui::Button("Reset Rotation"))
	{
		m_TransformToApply.SetRotation(glm::vec3(0.0f));
	}
	else
	{
		m_TransformToApply.Rotate(deltaQuat);
	}
}

void ModelLoaderWidget::DirectoryNagivation()
{

	DirectoryNavigatorHeader();


	if (ImGui::ListBoxHeader("", ImVec2(500.0f, 350.0f)))
	{
		fs::directory_entry selectedDir;
		std::map<std::string, fs::directory_entry> entries;
		for (auto dir : fs::directory_iterator(m_SelectedPath))
		{
			if (dir.is_directory())
				entries.emplace("[D]" + dir.path().filename().string(), dir);
			else
				entries.emplace("[F]" + dir.path().filename().string(), dir);
		}
		entries.emplace("..", m_SelectedPath.parent_path());

		for (auto entry : entries)
		{
			if (ImGui::MenuItem(entry.first.c_str()))
			{
				selectedDir = entry.second;
			}
		}

		if (selectedDir.is_directory())
		{
			if (IsDoubleClicked(selectedDir))
				m_SelectedPath = selectedDir.path();
		}
		else if (!selectedDir.path().empty())
		{
		    if (IsDoubleClicked(selectedDir))
		    {
				m_State = State::Loading;
				m_PathToOpen = selectedDir.path();
				m_AdditionalMessage = "Opening " + m_PathToOpen.filename().string() + ", this might take a few moments...";
		    }
		}

		if (selectedDir.exists())
		{
			m_FirstClick = selectedDir;
		}

		ImGui::ListBoxFooter();
	}
}

bool ModelLoaderWidget::IsDoubleClicked(std::filesystem::path a_Path)
{
	return m_FirstClick == a_Path;
}
