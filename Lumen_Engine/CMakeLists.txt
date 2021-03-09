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

set(DISABLE_CMAKE_SEARCH_PATHS ON)

set(ILMBASE_INCLUDEDIR "${CMAKE_CURRENT_SOURCE_DIR}/LumenPT/vendor/Include")
set(ILMBASE_LIBRARYDIR "${CMAKE_CURRENT_SOURCE_DIR}/LumenPT/vendor/Libs/OpenEXR")

set(TBB_INCLUDEDIR "${CMAKE_CURRENT_SOURCE_DIR}/LumenPT/vendor/Include")
set(TBB_LIBRARYDIR "${CMAKE_CURRENT_SOURCE_DIR}/LumenPT/vendor/Libs/tbb")

set(BLOSC_INCLUDEDIR "${CMAKE_CURRENT_SOURCE_DIR}/LumenPT/vendor/Include/blosc")
set(BLOSC_LIBRARYDIR "${CMAKE_CURRENT_SOURCE_DIR}/LumenPT/vendor/Libs")

set(ZLIB_ROOT 
"${CMAKE_CURRENT_SOURCE_DIR}/LumenPT/vendor/Include/zlib"
"${CMAKE_CURRENT_SOURCE_DIR}/LumenPT/vendor/Libs"
)

set(BOOST_INCLUDEDIR "${CMAKE_CURRENT_SOURCE_DIR}/LumenPT/vendor/Include")
set(BOOST_LIBRARYDIR "${CMAKE_CURRENT_SOURCE_DIR}/LumenPT/vendor/Libs/boost")

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