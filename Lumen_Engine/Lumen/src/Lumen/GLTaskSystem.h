#pragma once
#include <functional>
#include <thread>
#include <condition_variable>
#include <queue>


// Singleton class to allow moving all OpenGL calls to a single thread

class GLFWwindow;

class GLTaskSystem
{
public:
    static void Initialize(GLFWwindow* a_Window);

    //static uint32_t AddTask(std::function<void()> a_Task);

    static void WaitOnTask(uint32_t a_TaskID);

    static void Destroy();
private:

    GLTaskSystem(GLFWwindow* a_Window);
    ~GLTaskSystem();

    void StartProcessingThread(GLFWwindow* a_Window);

    uint32_t AddTaskImpl(std::function<void()> a_Task);

    void WaitOnTaskImpl(uint32_t a_TaskID);

    static GLTaskSystem* ms_Instance;

    struct Task
    {
        uint32_t m_ID;
        std::function<void()> m_Function;
    };

    std::queue<Task> m_Tasks;
    uint32_t m_IdCounter;
    std::unordered_map<uint32_t, std::condition_variable> m_TaskConditions;
    std::mutex m_TaskConditionsMutex;

    std::thread m_ProcessingThread;
    std::condition_variable m_ProcessingThreadCondition;
    std::mutex m_ProcessingThreadMutex;
    bool m_ProcessingThreadExit;
};
