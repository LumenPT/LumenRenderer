cmake_minimum_required (VERSION 3.16)

#project name (i.e. VS solution)
project ("Lumen_Engine" LANGUAGES C CXX)

add_compile_definitions(LMN_PLATFORM_WINDOWS)

#include sub-projects (i.e. VS projects)
	#GLFW
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

option(USE_WAVEFRONT "Use the Wavefront rendering pipeline?" ON)

add_subdirectory("Lumen/vendor/GLFW")
	#Glad
add_subdirectory ("Lumen/vendor/Glad")
	#ImGui
add_subdirectory ("Lumen/vendor/ImGui")

#set(SYSTEM_LIBRARY_PATHS "C:/Users/alexa/vcpkg/installed/x64-windows/include" "C:/Users/alexa/vcpkg/installed/x64-windows/lib" CACHE FILEPATH "Temp" FORCE)
#
#set(OPENVDB_DEPENDENCIES_INCLUDE CACHE FILEPATH "C:/Users/alexa/vcpkg/installed/x64-windows/include")
#set(OPENVDB_DEPENDENCIES_LIB CACHE FILEPATH "C:/Users/alexa/vcpkg/installed/x64-windows/lib")
#
#set(IlmBase_Half_LIBRARY "C:/Users/alexa/vcpkg/installed/x64-windows/include" CACHE FILEPATH "Temp" FORCE)
#set(IlmBase_INCLUDE_DIR "C:/Users/alexa/vcpkg/installed/x64-windows/lib" CACHE FILEPATH "Temp" FORCE)
#
#set(Tbb_INCLUDE_DIR "C:/Users/alexa/vcpkg/installed/x64-windows/include" CACHE FILEPATH "Temp" FORCE)
#set(Tbb_tbb_LIBRARY "C:/Users/alexa/vcpkg/installed/x64-windows/lib" CACHE FILEPATH "Temp" FORCE)
#set(Tbb_tbbmalloc_LIBRARY "C:/Users/alexa/vcpkg/installed/x64-windows/lib" CACHE FILEPATH "Temp" FORCE)
#
#set(Blosc_INCLUDE_DIR "C:/Users/alexa/vcpkg/installed/x64-windows/include" CACHE FILEPATH "Temp" FORCE)
#set(Blosc_LIBRARY "C:/Users/alexa/vcpkg/installed/x64-windows/lib" CACHE FILEPATH "Temp" FORCE)
#
#set(ZLIB_INCLUDE_DIR "C:/Users/alexa/vcpkg/installed/x64-windows/include" CACHE FILEPATH "Temp" FORCE)
#set(ZLIB_LIBRARY_DEBUG "C:/Users/alexa/vcpkg/installed/x64-windows/lib" CACHE FILEPATH "Temp" FORCE)
#set(ZLIB_LIBRARY_RELEASE "C:/Users/alexa/vcpkg/installed/x64-windows/lib" CACHE FILEPATH "Temp" FORCE)
#
#set(Boost_INCLUDE_DIR "C:/Users/alexa/vcpkg/installed/x64-windows/include" CACHE FILEPATH "Temp" FORCE)
#set(Boost_LIBRARY_DIR_DEBUG "C:/Users/alexa/vcpkg/installed/x64-windows/lib" CACHE FILEPATH "Temp" FORCE)
#set(Boost_LIBRARY_DIR_RELEASE "C:/Users/alexa/vcpkg/installed/x64-windows/lib" CACHE FILEPATH "Temp" FORCE)

#set(SYSTEM_LIBRARY_PATHS "C:/Users/alexa/vcpkg/installed/x64-windows/include" "C:/Users/alexa/vcpkg/installed/x64-windows/lib" CACHE FILEPATH "Temp" FORCE)

#set(OPENVDB_DEPENDENCIES_INCLUDE CACHE FILEPATH "C:/Users/alexa/vcpkg/installed/x64-windows/include")
#set(OPENVDB_DEPENDENCIES_LIB CACHE FILEPATH "C:/Users/alexa/vcpkg/installed/x64-windows/lib")

#set(IlmBase_Half_LIBRARY "D:/Media/Documents (D)/Game Development/Projects/LumenRenderer/Lumen_Engine/LumenPT/vendor/Include" CACHE STRING "Temp" FORCE)
set(IlmBase_Half_LIBRARY "${CMAKE_SOURCE_DIRECTORY}/LumenPT/vendor/Include" CACHE STRING "Temp" FORCE)
message("IlmBase_Half_LIBRARY file is ") 
message(${IlmBase_Half_LIBRARY})
set(IlmBase_INCLUDE_DIR "D:/Media/Documents (D)/Game Development/Projects/LumenRenderer/Lumen_Engine/LumenPT/vendor/Libs/OpenEXR" CACHE STRING "Temp" FORCE)

set(SMURF "smurf")
message("test")
message("${CMAKE_CURRENT_SOURCE_DIR}")
set(TBB_INCLUDEDIR "${CMAKE_CURRENT_SOURCE_DIR}/LumenPT/vendor/Include")
#set(Tbb_INCLUDE_DIR "D:/Media/Documents (D)/Game Development/Projects/LumenRenderer/Lumen_Engine/LumenPT/vendor/Include" CACHE FILEPATH "Temp" FORCE)
#set(Tbb_tbb_LIBRARY "D:/Media/Documents (D)/Game Development/Projects/LumenRenderer/Lumen_Engine/LumenPT/vendor/Libs/tbb" CACHE FILEPATH "Temp" FORCE)
#set(Tbb_tbbmalloc_LIBRARY "D:/Media/Documents (D)/Game Development/Projects/LumenRenderer/Lumen_Engine/LumenPT/vendor/Libs/tbb" CACHE FILEPATH "Temp" FORCE)

set(TBB_LIBRARYDIR "D:/Media/Documents (D)/Game Development/Projects/LumenRenderer/Lumen_Engine/LumenPT/vendor/Libs/tbb")
#set(Tbb_tbbmalloc_LIBRARY "D:/Media/Documents (D)/Game Development/Projects/LumenRenderer/Lumen_Engine/LumenPT/vendor/Libs/tbb" CACHE FILEPATH "Temp" FORCE)

set(Blosc_INCLUDE_DIR "D:/Media/Documents (D)/Game Development/Projects/LumenRenderer/Lumen_Engine/LumenPT/vendor/Include/blosc" CACHE FILEPATH "Temp" FORCE)
set(BLOSC_LIBRARYDIR "D:/Media/Documents (D)/Game Development/Projects/LumenRenderer/Lumen_Engine/LumenPT/vendor/Libs")

set(ZLIB_INCLUDE_DIR "D:/Media/Documents (D)/Game Development/Projects/LumenRenderer/Lumen_Engine/LumenPT/vendor/Include/zlib" CACHE FILEPATH "Temp" FORCE)
set(ZLIB_LIBRARY_DEBUG "D:/Media/Documents (D)/Game Development/Projects/LumenRenderer/Lumen_Engine/LumenPT/vendor/Libs" CACHE FILEPATH "Temp" FORCE)
set(ZLIB_LIBRARY_RELEASE "D:/Media/Documents (D)/Game Development/Projects/LumenRenderer/Lumen_Engine/LumenPT/vendor/Libs" CACHE FILEPATH "Temp" FORCE)

set(Boost_INCLUDE_DIR "D:/Media/Documents (D)/Game Development/Projects/LumenRenderer/Lumen_Engine/LumenPT/vendor/Include" CACHE FILEPATH "Temp" FORCE)
set(Boost_LIBRARY_DIR_DEBUG "D:/Media/Documents (D)/Game Development/Projects/LumenRenderer/Lumen_Engine/LumenPT/vendor/Libs/boost" CACHE FILEPATH "Temp" FORCE)
set(Boost_LIBRARY_DIR_RELEASE "D:/Media/Documents (D)/Game Development/Projects/LumenRenderer/Lumen_Engine/LumenPT/vendor/Libs/boost" CACHE FILEPATH "Temp" FORCE)

	#OpenVDB
add_subdirectory ("LumenPT/vendor/openvdb")
	#NanoVDB
add_subdirectory ("LumenPT/vendor/openvdb/nanovdb")
	#Lumen
add_subdirectory ("Lumen")
	#LumenPT
add_subdirectory ("LumenPT")
	#Sandbox
add_subdirectory ("Sandbox")

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT "Sandbox")	