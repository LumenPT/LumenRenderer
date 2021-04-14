#pragma once
#include <filesystem>
#include "Lumen/ModelLoading/SceneManager.h"

#include "Lumen/ModelLoading/Transform.h"

#include <thread>

class ModelLoaderWidget
{
public:

    ModelLoaderWidget(Lumen::SceneManager& a_SceneManager, std::shared_ptr<Lumen::ILumenScene>& a_SceneRef);

    ~ModelLoaderWidget();

    // Display the tool in a separate window
    void Display();

private:

    void LoadIcons();
    GLuint MakeGLTexture(std::string a_FilePath);

    // Enum describing the different states the widget can be in
    enum class State
    {
        Directory,
        Loading,
        ModelLoaded
    } m_State;

    // Enum describing the file that is being/has been loaded
    enum class FileType
    {
        Directory,
        GLTF,
        VDB
    };


    // Navigates the directory, mimicking windows explorer
    void DirectoryNagivation();
    // Header for the directory navigation, again mimicking windows explorer
    void DirectoryNavigatorHeader();

    // Base function called when a model is being loaded
    void LoadModel();

    // Base function to select what to do with a loaded model after it has been loaded
    void ModelSelection();

    // Use the loaded resource as a GLTF
    void ModelSelectionGLTF();

    // Helper function which inserts ImGui fields to specify a TRS transform, with rotation represented by euler angles
    void TransformSpecifier(Lumen::Transform& a_Transform, bool& a_ResetAfterApplied);

    bool IsDoubleClicked(std::filesystem::path a_Path);    

    // Reference to the scene that the renderer is using
    std::shared_ptr<Lumen::ILumenScene>& m_SceneRef;

    // The path that was last looked at in the directory navigation
    std::filesystem::path m_SelectedPath;
    // The path that was last clicked in directory navigation. Reset to empty if a double click passes through.
    std::filesystem::path m_FirstClick;

    Lumen::SceneManager& m_SceneManager;
    // Specification what file type was loaded
    FileType m_LoadedFileType;
    // The loaded GLTF resource.
    Lumen::SceneManager::GLTFResource* m_LoadedResource;
    // The transform that should be used when adding a mesh to the existing render scene
    Lumen::Transform m_TransformToApply;
    bool m_ResetTransformOnMeshAdded;
    // Path that needs to be opened for the file
    std::filesystem::path m_PathToOpen;

    // Message which serves as a way for the different states to communicate with the user.
    // Mostly used for confirmation and error messages
    std::string m_AdditionalMessage;

    // Handle to the thread that is used when loading a file. Used to avoid freezing the application while loading big files.
    std::thread m_LoadingThread;
    bool m_LoadingFinished;

    std::map<FileType, GLuint> m_Icons;



};

