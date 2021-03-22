#pragma once
#include "Lumen/Core.h"

namespace Lumen
{
	class Input
	{
	public:

		static void Update();

		inline static bool IsKeyPressed(int keycode){ return s_Instance->IsKeyPressedImpl(keycode); }
		inline static bool IsMouseButtonPressed(int button){ return s_Instance->IsMouseButtonPressedImpl(button); }

		inline static std::pair<float,float> GetMousePosition() { return m_CurrMousePos; }

		inline static float GetMouseX() { return m_CurrMousePos.first; }
		inline static float GetMouseY() { return m_CurrMousePos.second; }

		inline static std::pair<float, float> GetMouseDelta();
		inline static float GetMouseDeltaX();
		inline static float GetMouseDeltaY();

	protected:
		virtual bool IsKeyPressedImpl(int keycode) = 0;
		virtual bool IsMouseButtonPressedImpl(int button) = 0;
		
		virtual std::pair<float, float>  GetMousePositionImpl() = 0;
		
		virtual float GetMouseXImpl() = 0;
		virtual float GetMouseYImpl() = 0;

	private:
		static Input* s_Instance;
		inline static std::pair<float, float> m_CurrMousePos = { 0.0f, 0.0f };
		inline static std::pair<float, float> m_OldMousePos = { 0.0f, 0.0f };
	};

    inline void Input::Update()
    {
		m_OldMousePos = m_CurrMousePos;
		m_CurrMousePos = s_Instance->GetMousePositionImpl();
    }

    inline std::pair<float, float> Input::GetMouseDelta()
    {
		return { GetMouseDeltaX(), GetMouseDeltaY() };
    }

    inline float Input::GetMouseDeltaX()
    {
		return m_OldMousePos.first - GetMouseX();
    }

    inline float Input::GetMouseDeltaY()
    {
		return m_OldMousePos.second - GetMouseY();
    }
}

