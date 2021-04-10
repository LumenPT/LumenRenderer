#pragma once
#include <filesystem>
#include "Lumen/ModelLoading/SceneManager.h"

#include "Lumen/ModelLoading/Transform.h"

class ModelLoaderWidget
{
public:

    ModelLoaderWidget(Lumen::SceneManager& a_SceneManager, std::shared_ptr<Lumen::ILumenScene>& a_SceneRef);

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
    void TransformSpecifier();

    bool IsDoubleClicked(std::filesystem::path a_Path);    

    std::shared_ptr<Lumen::ILumenScene>& m_SceneRef;

    std::filesystem::path m_SelectedPath;
    std::filesystem::path m_FirstClick;

    Lumen::SceneManager& m_SceneManager;
    Lumen::SceneManager::GLTFResource* m_LoadedResource;
    Lumen::Transform m_TransformToApply;
    bool m_ResetTransformOnMeshAdded;
    std::filesystem::path m_PathToOpen;
    std::string m_AdditionalMessage;
};

