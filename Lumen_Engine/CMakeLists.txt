cmake_minimum_required (VERSION 3.20)

#project name (i.e. VS solution)
project ("Lumen_Engine" LANGUAGES C CXX)

add_compile_definitions(LMN_PLATFORM_WINDOWS)

#include sub-projects (i.e. VS projects)
	#GLFW
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

#Add option to use wavefront
#option(USE_WAVEFRONT "Use the Wavefront rendering pipeline?" ON)
set(USE_WAVEFRONT ON)
option(USE_DENOISER "Use Nvidia denoiser (if available)?" OFF)
option(USE_UPSCALING "Use Nvidia denoiser (if available)?" OFF)
option(INCLUDE_TESTPROJECT "Include test project" OFF)

SET(USE_NVIDIA_DENOISER FALSE)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/LumenPT/vendor/RayTracingDenoiser" AND USE_DENOISER)
SET(USE_NVIDIA_DENOISER TRUE)
endif()

SET(USE_NVIDIA_DLSS FALSE)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/LumenPT/vendor/DLSS_Private" AND USE_UPSCALING)
SET(USE_NVIDIA_DLSS TRUE)
endif()

#If wavefront is used, add compile definition.
if(${USE_WAVEFRONT})
	add_compile_definitions(WAVEFRONT)
endif()

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

#Set config name variable to use for different IDEs
set(CONFIG_NAME "")

if(	${CMAKE_GENERATOR} STREQUAL "Visual Studio 8 2005" OR
	${CMAKE_GENERATOR} STREQUAL "Visual Studio 9 2008")

	set(CONFIG_NAME $(ConfigurationName))

elseif(
	${CMAKE_GENERATOR} STREQUAL "Visual Studio 10 2010" OR
	${CMAKE_GENERATOR} STREQUAL "Visual Studio 11 2012" OR
	${CMAKE_GENERATOR} STREQUAL "Visual Studio 12 2013" OR
	${CMAKE_GENERATOR} STREQUAL "Visual Studio 14 2015" OR
	${CMAKE_GENERATOR} STREQUAL "Visual Studio 15 2017" OR
	${CMAKE_GENERATOR} STREQUAL "Visual Studio 16 2019")

	set(CONFIG_NAME $(Configuration))

elseif(
	${CMAKE_GENERATOR} STREQUAL "Xcode" OR
	${CMAKE_GENERATOR} STREQUAL "Ninja Multi-Config")

	set(CONFIG_NAME $(CONFIGURATION))
	
endif()

#Add config name variable as compile definition.
add_compile_definitions(CONFIG_NAME=\"${CONFIG_NAME}\")
	#OpenVDB
add_subdirectory ("LumenPT/vendor/openvdb")
	#NanoVDB
add_subdirectory ("LumenPT/vendor/openvdb/nanovdb")
	#Lumen
add_subdirectory ("Lumen")
	#LumenPT
add_subdirectory ("LumenPT")
if(INCLUDE_TESTPROJECT)
	add_subdirectory ("TestProject")
endif()
	#Sandbox
add_subdirectory ("Sandbox")

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT "Sandbox")