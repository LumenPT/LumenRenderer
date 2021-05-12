#include "GLTaskSystem.h"

#include "glad/glad.h"
#include "GLFW/glfw3.h"

GLTaskSystem* GLTaskSystem::ms_Instance = nullptr;

// Standard singleton fluff
void GLTaskSystem::Initialize(GLFWwindow* a_Window)
{
    ms_Instance = new GLTaskSystem(a_Window);
}
//
//uint32_t GLTaskSystem::AddTask(std::function<void()> a_Task)
//{
//    return ms_Instance->AddTaskImpl(a_Task);
//}

void GLTaskSystem::WaitOnTask(uint32_t a_TaskID)
{
    ms_Instance->WaitOnTaskImpl(a_TaskID);
}

void GLTaskSystem::Destroy()
{
    delete ms_Instance;
}

GLTaskSystem::GLTaskSystem(GLFWwindow* a_Window)
    : m_ProcessingThreadExit(false)
    , m_IdCounter(0)
{
    StartProcessingThread(a_Window);
}

GLTaskSystem::~GLTaskSystem()
{
    // Tell the processing thread to exit when it finishes its current operations
    m_ProcessingThreadExit = true;
    while (m_Tasks.size())
    {
        m_Tasks.pop();
    }

    m_ProcessingThreadCondition.notify_all();
    // Wait for the processing thread to finish up
    m_ProcessingThread.join();
}

void GLTaskSystem::StartProcessingThread(GLFWwindow* a_Window)
{
    
    // Start a thread running the defined lambda function
    m_ProcessingThread = std::thread([this, a_Window]()
        {
            glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
            GLFWwindow* threadWin = glfwCreateWindow(1, 1, "Thread Window", NULL, NULL);
            auto err1 = glGetError();

            // Have the thread run as long as it's not told to exit
            while (!m_ProcessingThreadExit)
            {
                if (m_Tasks.empty() && !m_ProcessingThreadExit)
                {
                    // Put the thread to sleep via the condition variable if there are no tasks for it
                    std::unique_lock lock(m_ProcessingThreadMutex);
                    m_ProcessingThreadCondition.wait(lock, [&]() 
                        {
                            return !m_Tasks.empty() || m_ProcessingThreadExit;
                        });
                }

                if (m_ProcessingThreadExit)
                    break;
                // Perform the task
                m_Tasks.front().m_Function();
                auto err = glGetError();
                glFlush();

                m_TaskConditionsMutex.lock();
                // Wake up any threads that were waiting on this task to be finished
                m_TaskConditions[m_Tasks.front().m_ID].notify_all();
                // Delete the condition variable that was associated with the task as it is no longer needed
                m_TaskConditions.erase(m_Tasks.front().m_ID);

                m_TaskConditionsMutex.unlock();

                // Task is not finished and can be deleted
                m_Tasks.pop();
            }
        });
}

uint32_t GLTaskSystem::AddTaskImpl(std::function<void()> a_Task)
{
    uint32_t id = m_IdCounter++;

    Task t;
    t.m_ID = id;
    t.m_Function = a_Task;

    m_Tasks.push(t);

    // Wake up the processing thread if it was sleeping due to lack of tasks
    m_ProcessingThreadCondition.notify_all();

    // Add a condition variable to the task, as it is used to determine if the task has already been finished
    m_TaskConditions[t.m_ID];

    // Return an ID that can be used to wait on the task's execution
    return id;
}

void GLTaskSystem::WaitOnTaskImpl(uint32_t a_TaskID)
{
    // Put the calling thread to sleep here, using the task ID's condition variable as a wake condition
    m_TaskConditionsMutex.lock();
    if (m_TaskConditions.find(a_TaskID) != m_TaskConditions.end())
    {
        std::mutex mutex;
        std::unique_lock lock(mutex);
        auto& condition = m_TaskConditions[a_TaskID];

        m_TaskConditionsMutex.unlock();
        condition.wait(lock);        
    }
    else
    {
        m_TaskConditionsMutex.unlock();
    }
}
