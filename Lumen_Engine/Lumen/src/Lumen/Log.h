#pragma once

#include "Core.h"
#include "spdlog/spdlog.h"
#include "spdlog/fmt/ostr.h"

namespace Lumen
{
	
	class Log
	{
	public:
		static void Init();

		inline static std::shared_ptr<spdlog::logger>& GetCoreLogger() { return s_CoreLogger; }
		inline static std::shared_ptr<spdlog::logger>& GetClientLogger() { return s_ClientLogger; }

	private:
		static std::shared_ptr<spdlog::logger> s_CoreLogger;
		static std::shared_ptr<spdlog::logger> s_ClientLogger;
		
	};

}

// Core log macros
#define LMN_CORE_TRACE(...) ::Lumen::Log::GetCoreLogger()->trace(__VA_ARGS__)
#define LMN_CORE_INFO(...)  ::Lumen::Log::GetCoreLogger()->info(__VA_ARGS__)
#define LMN_CORE_WARN(...)  ::Lumen::Log::GetCoreLogger()->warn(__VA_ARGS__)
#define LMN_CORE_ERROR(...) ::Lumen::Log::GetCoreLogger()->error(__VA_ARGS__)
#define LMN_CORE_FATAL(...) ::Lumen::Log::GetCoreLogger()->fatal(__VA_ARGS__)

// Client log macros
#define LMN_TRACE(...)      ::Lumen::Log::GetClientLogger()->trace(__VA_ARGS__)
#define LMN_INFO(...)       ::Lumen::Log::GetClientLogger()->info(__VA_ARGS__)
#define LMN_WARN(...)       ::Lumen::Log::GetClientLogger()->warn(__VA_ARGS__)
#define LMN_ERROR(...)      ::Lumen::Log::GetClientLogger()->error(__VA_ARGS__)
#define LMN_FATAL(...)      ::Lumen::Log::GetClientLogger()->fatal(__VA_ARGS__)
