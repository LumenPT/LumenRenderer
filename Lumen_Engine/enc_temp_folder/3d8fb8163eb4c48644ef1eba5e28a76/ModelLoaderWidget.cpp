#include "ModelLoaderWidget.h"

#include "ImGui/imgui.h"

#include <map>

namespace fs = std::filesystem;

ModelLoaderWidget::ModelLoaderWidget()
{
	m_SelectedPath = fs::current_path();
}

void ModelLoaderWidget::Display()
{
	ImGui::SetNextWindowSize(ImVec2(550.0f, 600.0f));
	ImGui::Begin("Testing");

	DirectoryNavigator();


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
		else if (selectedDir.path().extension() == ".gltf")
		{
            if (IsDoubleClicked(selectedDir))
			    printf("Load this shid, GLTF edition\n");                
		}
		else if (selectedDir.path().extension() == ".vdb")
		{
			if (IsDoubleClicked(selectedDir))
			    printf("Load this shid, VDB edition\n");
		}

        if (selectedDir.exists())
        {
		    m_FirstClick = selectedDir;            
        }

		ImGui::ListBoxFooter();
	}

	ImGui::End();
}

void ModelLoaderWidget::DirectoryNavigator()
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

bool ModelLoaderWidget::IsDoubleClicked(std::filesystem::path a_Path)
{
	return m_FirstClick == a_Path;
}
