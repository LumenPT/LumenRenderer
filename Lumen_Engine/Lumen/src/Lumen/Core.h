#pragma once

#ifdef LMN_PLATFORM_WINDOWS
#if LMN_DYNAMIC_LINK
	#ifdef LMN_BUILD_DLL
		#define LUMEN_API __declspec(dllexport)
	#else
		#define LUMEN_API __declspec(dllimport)
	#endif
#else
#define LUMEN_API
#endif
#else
	#error Lumen only supports windows!
#endif

#ifdef LMN_DEBUG
	#define LMN_ENABLE_ASSERTS
#endif

#ifdef LMN_ENABLE_ASSERTS
	#define LMN_ASSERT(x, ...) { if(!(x)) {LMN_ERROR("Assertion Failed: {0}", __VA_ARGS__); __debugbreak(); } }
	#define LMN_CORE_ASSERT(x, ...) { if(!(x)) {LMN_CORE_ERROR("Assertion Failed: {0}", __VA_ARGS__); __debugbreak(); } } 
#else
	#define LMN_ASSERT(x, ...)
	#define LMN_CORE_ASSERT(x, ...)
#endif

#define BIT(x) (1 << x)

#define LMN_BIND_EVENT_FN(fn) std::bind(&fn, this, std::placeholders::_1)