#pragma once
#include "lmnpch.h"

#include "Lumen/Core.h"
#include "Lumen/Events/Event.h"

namespace Lumen
{
	/// <summary>
	/// Generic window data
	/// </summary>
	struct WindowProps
	{
		std::string title;
		unsigned int width;
		unsigned int height;

		WindowProps(const std::string& a_Title = "Lumen Rendering Engine",
			unsigned int a_Width = 1280,
			unsigned int a_Height = 720)
			: title(a_Title), width(a_Width), height(a_Height)
		{
		}
	};

	// interface representing a desktop system based window
	class Window
	{
	public:
		using EventCallbackFn = std::function<void(Event&)>;

		virtual ~Window() {};

		virtual void OnUpdate() = 0;

		virtual unsigned int GetWidth() const = 0;
		virtual unsigned int GetHeight() const = 0;

		//window attributes
		virtual void SetEventCallback(const EventCallbackFn& callback) = 0;
		virtual void SetVSync(bool enabled) = 0;
		virtual bool IsVSync() const = 0;

		virtual void* GetNativeWindow() const = 0;
		
		static Window* Create(const WindowProps& props = WindowProps());
	};
	
	
}