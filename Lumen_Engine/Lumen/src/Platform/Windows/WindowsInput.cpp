#include "lmnpch.h"
#include "WindowsInput.h"
#include "Lumen/LumenApp.h"

#include <GLFW/glfw3.h>

namespace Lumen
{
	Input* Input::s_Instance = new WindowsInput();

    void WindowsInput::ScrollCallBack(GLFWwindow*, double a_XOffset, double a_YOffset)
    {
		m_MouseWheelX = a_XOffset;
		m_MouseWheelY = a_YOffset;
    }

    void WindowsInput::SetCallbacksImpl()
    {
		glfwSetScrollCallback(reinterpret_cast<GLFWwindow*>(LumenApp::Get().GetWindow().GetNativeWindow()), &ScrollCallBack);
    }

    bool WindowsInput::IsKeyPressedImpl(int keycode)
	{
		auto window = static_cast<GLFWwindow*>(LumenApp::Get().GetWindow().GetNativeWindow());
		auto state = glfwGetKey(window, keycode);

		return state == GLFW_PRESS || state == GLFW_REPEAT;
	}

	bool WindowsInput::IsMouseButtonPressedImpl(int button)
	{
		auto window = static_cast<GLFWwindow*>(LumenApp::Get().GetWindow().GetNativeWindow());
		auto state = glfwGetMouseButton(window, button);

		return state == GLFW_PRESS;
	}

	std::pair<float, float> WindowsInput::GetMousePositionImpl()
	{
		auto window = static_cast<GLFWwindow*>(LumenApp::Get().GetWindow().GetNativeWindow());
		double xpos, ypos;
		glfwGetCursorPos(window, &xpos, &ypos);

		return { static_cast<float>(xpos), static_cast<float>(ypos) };
	}

	float WindowsInput::GetMouseXImpl()
	{
		auto [x, y] = GetMousePositionImpl();
		return x;
	}

	float WindowsInput::GetMouseYImpl()
	{
		auto [x, y] = GetMousePositionImpl();
		return y;
	}

}



