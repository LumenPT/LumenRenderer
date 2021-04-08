#pragma once
#include <filesystem>

class ModelLoaderWidget
{
public:

    ModelLoaderWidget();

    void Display();

private:

    void DirectoryNavigator();

    bool IsDoubleClicked(std::filesystem::path a_Path);

    std::filesystem::path m_SelectedPath;
    std::filesystem::path m_FirstClick;
};

