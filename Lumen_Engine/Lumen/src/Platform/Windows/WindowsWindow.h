#pragma once

#include "Lumen/Window.h"

#include <GLFW/glfw3.h>

struct GLFWwindow;

namespace Lumen
{
	class WindowsWindow : public Window
	{
	public:
		WindowsWindow(const WindowProps& props);
		virtual ~WindowsWindow();

		void OnUpdate() override;

		inline unsigned int GetWidth() const override { return m_Data.width; };
		inline unsigned int GetHeight() const override { return m_Data.height; };

		// Window Attributes
		inline void SetEventCallback(const EventCallbackFn& callback) override { m_Data.EventCallback = callback; };
		void SetVSync(bool enabled) override;
		bool IsVSync() const override;

		inline virtual void* GetNativeWindow() const { return m_Window; }
		
	private:
		virtual void Init(const WindowProps& props);
		virtual void Shutdown();
		//
		GLFWwindow* m_Window;

		struct WindowData
		{
			std::string title = "";
			unsigned int width = 0, height = 0;
			bool vSync = false;

			EventCallbackFn EventCallback;
		};

		WindowData m_Data;
		
	};
}
