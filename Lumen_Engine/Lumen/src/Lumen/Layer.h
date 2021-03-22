#pragma once

#include "Lumen/Core.h"
#include "Lumen/Events/Event.h"
#include "Lumen/LumenApp.h"

namespace Lumen
{
	class Layer
	{
		friend class LumenApp; // For LumenApp to be able to access SetLayerServices()
	public:
		Layer(const std::string& name = "Layer");
		virtual ~Layer();

		virtual void OnAttach() {}
		virtual void OnDetach() {}
		virtual void OnUpdate() {}
		virtual void OnDraw() {}
		virtual void OnImGuiRender() {}
		virtual void OnEvent(Event& event) {}

		inline const std::string& GetName() const { return m_DebugName; }
        
	protected:
		std::string m_DebugName;

		LumenApp::LayerServices* m_LayerServices;

	private:
		void SetLayerServices(LumenApp::LayerServices* a_ServicesPtr) { m_LayerServices = a_ServicesPtr; }
	};
};
