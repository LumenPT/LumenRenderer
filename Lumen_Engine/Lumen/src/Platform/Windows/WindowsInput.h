#pragma once

#include "Lumen/Input.h"
class GLFWwindow;
namespace Lumen
{
	class WindowsInput : public Input
	{
	public:

		static void ScrollCallBack(GLFWwindow* a_Window, double a_XOffset, double a_YOffset);

	protected:
		void SetCallbacksImpl() override;

		virtual bool IsKeyPressedImpl(int keycode) override;

		virtual bool IsMouseButtonPressedImpl(int button) override;

		virtual std::pair<float, float>  GetMousePositionImpl() override;
		
		virtual float GetMouseXImpl() override;
		virtual float GetMouseYImpl() override;
	};
} 