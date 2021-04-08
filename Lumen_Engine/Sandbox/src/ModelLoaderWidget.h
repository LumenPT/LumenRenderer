#pragma once
#include <filesystem>
#include "Lumen/ModelLoading/SceneManager.h"

class ModelLoaderWidget
{
public:

    ModelLoaderWidget(Lumen::SceneManager& a_SceneManager);

    void Display();

private:

    enum class State
    {
        Directory,
        Loading,
        ModelLoaded
    } m_State;

    void DirectoryNagivation();
    void DirectoryNavigatorHeader();

    void LoadModel();

    void ModelSelection();

    bool IsDoubleClicked(std::filesystem::path a_Path);    

    std::filesystem::path m_SelectedPath;
    std::filesystem::path m_FirstClick;

    Lumen::SceneManager& m_SceneManager;
    Lumen::SceneManager::GLTFResource* m_LoadedResource;
    std::filesystem::path m_PathToOpen;
    std::string m_AdditionalMessage;
};

