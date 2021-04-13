#include "ModelLoaderWidget.h"

#include "ImGui/imgui.h"

#include "Lumen/../AssetLoading/AssetLoading.h"

#include "Tools/ImGuiUtil.h"

#include <map>

namespace fs = std::filesystem;

ModelLoaderWidget::ModelLoaderWidget(Lumen::SceneManager& a_SceneManager, std::shared_ptr<Lumen::ILumenScene>& a_SceneRef)
    : m_SceneManager(a_SceneManager)
    , m_SceneRef(a_SceneRef)
    , m_State(State::Directory)
    , m_LoadingFinished(true)
{
	LoadIcons();
	// Initialze the selected path to the working directory for the application
	m_SelectedPath = fs::current_path();
}

ModelLoaderWidget::~ModelLoaderWidget()
{
	// Thread needs to be joined if it was used before, otherwise it will throw an error
	if (m_LoadingThread.joinable())
		m_LoadingThread.join();

}

void ModelLoaderWidget::Display()
{
	// TODO: Probably better to not hardcode the window's size, but then we need better other stuff
	ImGui::SetNextWindowSize(ImVec2(550.0f, 600.0f));
	ImGui::Begin("Model and File loader");

	// State machine to determine the display of the widget
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

	// Display the additional message the program has
	if (!m_AdditionalMessage.empty())
		ImGui::Text(m_AdditionalMessage.c_str());

	ImGui::End();
}

void ModelLoaderWidget::LoadIcons()
{
	auto p = fs::current_path();

    while (p.filename() != "Lumen_Engine")
    {
		p = p.parent_path();
    }

	fs::current_path(p);

	m_Icons[FileType::Directory] = MakeGLTexture("Sandbox/assets/toolAssets/folder.png");
	m_Icons[FileType::GLTF] = MakeGLTexture("Sandbox/assets/toolAssets/file.png");
	m_Icons[FileType::VDB] = MakeGLTexture("Sandbox/assets/toolAssets/file.png");
}

GLuint ModelLoaderWidget::MakeGLTexture(std::string a_FilePath)
{

	int w, h, c;
	auto imgData = stbi_load(a_FilePath.c_str(), &w, &h, &c, 4);

	GLuint tex;
	glGenTextures(1, &tex);

	glBindTexture(GL_TEXTURE_2D, tex);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, imgData);

	// Set the sampler settings, standard procedure
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	// Unbind the pixel buffer and texture to avoid mistakes with future OpenGL calls
	glBindTexture(GL_TEXTURE_2D, 0);

	stbi_image_free(imgData);

	return tex;
}

void ModelLoaderWidget::DirectoryNavigatorHeader()
{
	struct Dir
	{
		std::string m_Name;
		fs::path m_Path;
	};
	// Get all individual valid paths that are part of the currently selected path
	std::vector<Dir> dirs;
	auto p = m_SelectedPath;
    do 
    {
		Dir d = { p.filename().string(), p };
		dirs.push_back(d);
		p = p.parent_path();
	} while (p.has_relative_path());

	// Also add a path for the root directory
	Dir d = { p.string(), p };
	dirs.push_back(d);

	// Create a button for each path
	// The root directory path is the last in the list to be added, so we iterate in reverse
    for (int64_t i = dirs.size() - 1; i >= 0; --i)
    {
		auto& dir = dirs[i];
		if (ImGui::Button(dir.m_Name.c_str()))
			m_SelectedPath = dir.m_Path;

		// If 'i' is 0, this is the last button to add, so we don't need to add ImGui::SameLine()
        if (i)
		    ImGui::SameLine();            
    }
}

void ModelLoaderWidget::LoadModel()
{
    if (m_LoadingFinished)
    {
		// Thread needs to be joined if it was used before, otherwise it will throw an error
        if (m_LoadingThread.joinable())
		    m_LoadingThread.join();
		m_LoadingThread = std::thread([this]() {
			m_LoadingFinished = false;
			// Choose how to open the file based on its extension
			if (m_PathToOpen.extension() == ".gltf")
			{
				
				m_LoadedResource = m_SceneManager.LoadGLTF(m_PathToOpen.filename().string(), m_PathToOpen.parent_path().string() + "\\");
				m_State = State::ModelLoaded;
				m_AdditionalMessage = "";
				m_LoadedFileType = FileType::GLTF;
			}
			else
			{
				m_AdditionalMessage = "Loading of " + m_PathToOpen.extension().string() + " files is not supported yet.";
				m_State = State::Directory;
			}
			m_LoadingFinished = true;
		});
	}
}

void ModelLoaderWidget::ModelSelection()
{
	ImGui::Text("File Loaded Successfully. What would you like to do with it?");

    switch (m_LoadedFileType)
    {
	case FileType::GLTF:
		ModelSelectionGLTF();
		break;
	default:
		m_AdditionalMessage = "No response has been implemented for this file type. See ModelLoaderWidget::ModelSelection().";
    }

	if (ImGui::Button("Return to directory view"))
	{
		m_State = State::Directory;
		m_LoadedResource = nullptr;
	}
}

void ModelLoaderWidget::ModelSelectionGLTF()
{
	ImGui::Text("Loaded file is .gltf.\nYou can replace the current scene with one from the file,\nor extract individual meshes from the file into the scene");

	// If the file has any scenes specified, display them as possible scene options
	if (!m_LoadedResource->m_Scenes.empty() && ImGui::ListBoxHeader("Loaded scenes"))
	{
		// The menu size is needed to offset the "Add Scene" button correctly
		auto menuRect = ImGui::GetItemRectSize();
		for (size_t i = 0; i < m_LoadedResource->m_Scenes.size(); i++)
		{
			// Display the name of the scene. If none was specified in the file, the scene loader default to "Unnamed Scene X"
			ImGui::Text(m_LoadedResource->m_Scenes[i]->m_Name.c_str());

			auto rMin = ImGui::GetItemRectMin();
			auto rMax = ImGui::GetItemRectMax();

			ImVec2 min = rMin;
			ImVec2 max = ImVec2(rMin.x + menuRect.x, rMax.y + 5);

			// Is the mouse cursor hovering over the scene's entry in the menu
			if (ImGui::IsMouseHoveringRect(min, max))
			{
				// Button is on the same line, and 80 units away from the right border of the menu
				ImGui::SameLine();
				ImGui::SetCursorPosX(menuRect.x - 80.0f);
				// Force the item width to also be 80 units
				ImGui::PushItemWidth(80.0f);
				// Create the button to set the scene, and if it is clicked change the scene the renderer is using
				if (ImGui::Button("Set scene"))
				{
					m_SceneRef = m_LoadedResource->m_Scenes[i];
					m_AdditionalMessage = "Scene successfully set to " + m_LoadedResource->m_Scenes[i]->m_Name;
				}
			}
		}
		ImGui::ListBoxFooter();
	}


	// Add a transform tool to determine the initial transform of the mesh that can be added to the scene
	TransformSpecifier(m_TransformToApply, m_ResetTransformOnMeshAdded);

	// If the file has meshes, make a menu for them
	if (!m_LoadedResource->m_MeshPool.empty() && ImGui::ListBoxHeader("Loaded Meshes"))
	{
		// Needed to determine the offset for the "Add Mesh" button
		auto menuRect = ImGui::GetItemRectSize();

		for (size_t i = 0; i < m_LoadedResource->m_MeshPool.size(); i++)
		{
			// GLTF does not name its meshes, so we can only use numbers as identifiers
			ImGui::Text("Mesh %llu", i);

			auto rMin = ImGui::GetItemRectMin();
			auto rMax = ImGui::GetItemRectMax();

			ImVec2 min = rMin;
			ImVec2 max = ImVec2(rMin.x + menuRect.x, rMax.y + 5);

			// Is the mesh entry hovered on by the mouse cursor?
			if (ImGui::IsMouseHoveringRect(min, max))
			{
				// Add a button on the same line, but on the opposite side of the menu
				ImGui::SameLine();
				ImGui::SetCursorPosX(menuRect.x - 80.0f);
				ImGui::PushItemWidth(80.0f); // Force the button size to 80 units
				if (ImGui::Button("Add Mesh"))
				{
					// If the buttow was clicked, add a mesh to the scene and assign it the current mesh
					// and the specified transform
					auto m = m_SceneRef->AddMesh();

					m->SetMesh(m_LoadedResource->m_MeshPool[i]);
					m->m_Transform = m_TransformToApply;
					if (m_ResetTransformOnMeshAdded) // Reset the transform if requested by the user
						m_TransformToApply = Lumen::Transform();

					m_AdditionalMessage = "Mesh successfully added to the scene.";
				}
			}
		}
		ImGui::ListBoxFooter();
	}

}


void ModelLoaderWidget::TransformSpecifier(Lumen::Transform& a_Transform, bool& a_ResetAfterApplied)
{
	glm::vec3 t, r, s;
	t = a_Transform.GetPosition();
	r = a_Transform.GetRotationEuler();
	s = a_Transform.GetScale();

	ImGui::Checkbox("Reset transform after adding mesh", &a_ResetAfterApplied);
	ImGui::DragFloat3("Position", &t[0]);
	ImGui::DragFloat3("Rotation", &r[0]);
	ImGui::DragFloat3("Scale", &s[0]);
	a_Transform.SetPosition(t);
	a_Transform.SetScale(s);

	// Quaternion magic, essentially
	// We calculate the difference in euler angles between the requested rotation and the previous rotation.
	// We then create a quaternion from that delta and rotate the current rotation with it.
	// This avoids the quaternion from spazzing out when the Y rotation is at 90 degrees.
	auto deltaRotation = r - a_Transform.GetRotationEuler();

	glm::quat deltaQuat = glm::quat(glm::radians(deltaRotation));
	a_Transform.Rotate(deltaQuat);
}

void ModelLoaderWidget::DirectoryNagivation()
{
	DirectoryNavigatorHeader();

	// The directory navigation
	if (ImGui::ListBoxHeader("", ImVec2(500.0f, 350.0f)))
	{
		struct Dir
		{
			std::string m_Name;
			fs::directory_entry m_DirEntry;
			FileType m_Type;
		};


		fs::directory_entry selectedDir;
		std::vector<Dir> entries;
		// Go through all directories and files in order to mark them accordingly
		// Might replace with Icons later for +fancy points
		for (auto dir : fs::directory_iterator(m_SelectedPath))
		{
			Dir d;
			d.m_Name = dir.path().filename().string();
			
			if (dir.is_directory())
				//entries.emplace("[D]" + dir.path().filename().string(), { dir, FileType::Directory }); // Mark as directory
				d.m_Type = FileType::Directory;
			else
				d.m_Type = FileType::GLTF;

			d.m_DirEntry = dir;

			entries.push_back(d);
				//entries.emplace("[F]" + dir.path().filename().string(), {dir, FileType::GLTF}); // Mark as file
		}

		entries.push_back({ std::string(".."), fs::directory_entry(m_SelectedPath.parent_path()), FileType::Directory });
		std::sort(entries.begin(), entries.end(), [](const Dir& a_Left, const Dir& a_Right)
			{
				if (a_Left.m_Type == FileType::Directory && a_Left.m_Type != a_Right.m_Type)
				{
					return true;
				}

				auto length = std::max(a_Left.m_Name.size(), a_Right.m_Name.size());

				for (size_t i = 0; i < length; i++)
				{
					auto l = std::tolower(a_Left.m_Name[i]);
					auto r = std::tolower(a_Right.m_Name[i]);
					if (l != r)
					{
						return l < r;
					}
				}

				return a_Left.m_Name.size() < a_Right.m_Name.size();
			});

		// Add an old-school "go back" option to the list of directories

		// Make a menu item for each directory
		for (auto entry : entries)
		{
			ImGuiUtil::DisplayImage(m_Icons[entry.m_Type], glm::ivec2(8));
			ImGui::SameLine();

			if (ImGui::MenuItem(entry.m_Name.c_str()))
			{
				// set the selected dir to the current item's directory if it was clicked
				selectedDir = entry.m_DirEntry;
			}
		}

		// If the selected path is a directory, we go into it on double click
		if (selectedDir.is_directory())
		{
			if (IsDoubleClicked(selectedDir))
				m_SelectedPath = selectedDir.path();
		}
		else if (!selectedDir.path().empty())
		{
			// Otherwise attempt to open the path assuming it's not empty
		    if (IsDoubleClicked(selectedDir))
		    {
				// Enter the loading state
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
