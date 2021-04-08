#include "ModelLoaderWidget.h"

#include "ImGui/imgui.h"

#include <map>

namespace fs = std::filesystem;

ModelLoaderWidget::ModelLoaderWidget(Lumen::SceneManager& a_SceneManager)
    : m_SceneManager(a_SceneManager)
    , m_State(State::Directory)
{
	m_SelectedPath = fs::current_path();
}

void ModelLoaderWidget::Display()
{
	ImGui::SetNextWindowSize(ImVec2(550.0f, 600.0f));
	ImGui::Begin("Testing");

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
}

void ModelLoaderWidget::ModelSelection()
{
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
		else
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
