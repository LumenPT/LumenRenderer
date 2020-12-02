cmake_minimum_required (VERSION 3.16)

#project name (i.e. VS solution)
project ("Lumen_Engine")

add_compile_definitions(LMN_PLATFORM_WINDOWS)

#include sub-projects (i.e. VS projects)
	#GLFW
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory("Lumen/vendor/GLFW")
	#Glad
add_subdirectory ("Lumen/vendor/Glad")
	#ImGui
add_subdirectory ("Lumen/vendor/ImGui")
	#Lumen
add_subdirectory ("Lumen")
	#LumenPT
add_subdirectory ("LumenPT")
	#Sandbox
add_subdirectory ("Sandbox")