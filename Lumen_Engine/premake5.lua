workspace "Lumen_Engine"
	architecture "x64"
	startproject "Sandbox"
	
	configurations
	{
		"Debug",
		"Release",
		"Dist"
	}

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

-- include directories relative to root folder
IncludeDir = {}
IncludeDir["GLFW"] = "Lumen/vendor/GLFW/include"
IncludeDir["Glad"] = "Lumen/vendor/Glad/include"
IncludeDir["ImGui"] = "Lumen/vendor/imgui"
IncludeDir["glm"] = "Lumen/vendor/glm"

include "Lumen/vendor/GLFW"
include "Lumen/vendor/Glad"
include "Lumen/vendor/imgui"

project "Lumen"
	location "Lumen"
	kind "StaticLib"
	language "C++"
	cppdialect "C++17"
	staticruntime "on"

	targetdir ("bin/" .. outputdir .. "/%{prj.name}")
	objdir ("bin-int/" .. outputdir .. "/%{prj.name}")

	pchheader "lmnpch.h"
	pchsource "Lumen/src/lmnpch.cpp"

	files
	{
		"%{prj.name}/src/**.h",
		"%{prj.name}/src/**.cpp",
		"%{prj.name}/vendor/glm/glm/**.hpp",
		"%{prj.name}/vendor/glm/glm/**.inl"
	}

	defines
	{
		"_CRT_SECURE_NO_WARNINGS"
	}

	includedirs
	{
		"%{prj.name}/src",
		"%{prj.name}/vendor/spdlog/include",
		"%{IncludeDir.GLFW}",
		"%{IncludeDir.Glad}",
		"%{IncludeDir.ImGui}",
		"%{IncludeDir.glm}"
	}
	
	links
	{
		"GLFW",
		"Glad",
		"ImGui",
		"opengl32.lib"
	}

	filter "system:windows"
		systemversion "latest"

		defines
		{
			"LMN_PLATFORM_WINDOWS",
			"LMN_BUILD_DLL",
			"GLFW_INCLUDE_NONE"
		}

	filter "configurations:Debug"
		defines "LMN_DEBUG"
		runtime "Debug"
		symbols "on"
		
	filter "configurations:Release"
		defines "LMN_RELEASE"
		runtime "Release"
		optimize "on"
		
	filter "configurations:Dist"
		defines "LMN_DIST"
		runtime "Release"
		optimize "on"

project "Sandbox"
	location "Sandbox"
	kind "ConsoleApp"
	language "C++"
	cppdialect "C++17"
	staticruntime "on"

	targetdir ("bin/" .. outputdir .. "/%{prj.name}")
	objdir ("bin-int/" .. outputdir .. "/%{prj.name}")

	files
	{
		"%{prj.name}/src/**.h",
		"%{prj.name}/src/**.cpp"
	}

	includedirs 
	{
		"Lumen/vendor/spdlog/include",
		"Lumen/src",
		"Lumen/vendor",
		"%{IncludeDir.glm}"
	}

	links
	{
		"Lumen"
	}
	
	filter "system:windows"
		systemversion "latest"

		defines
		{
			"LMN_PLATFORM_WINDOWS"
		}

	filter "configurations:Debug"
		defines "LMN_DEBUG"
		runtime "Debug"
		symbols "on"
		
	filter "configurations:Release"
		defines "LMN_RELEASE"
		runtime "Release"
		optimize "on"
		
	filter "configurations:Dist"
		defines "LMN_DIST"
		runtime "Release"
		optimize "on"