cmake_minimum_required (VERSION 3.20)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#Add header files
file(GLOB_RECURSE HEADER_FILES 

	${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp
)

#Add source files
file(GLOB_RECURSE SOURCE_FILES 

	${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

# Add header and source files to library
add_library (Lumen STATIC ${HEADER_FILES} ${SOURCE_FILES})
set_property(TARGET Lumen APPEND PROPERTY COMPILE_DEFINITIONS LUMEN)

include(${CMAKE_SOURCE_DIR}/cmake/functions.cmake)
assign_source_group(${HEADER_FILES})
assign_source_group(${SOURCE_FILES})

#add precompiled header file
target_precompile_headers(Lumen
	PRIVATE
		"src/lmnpch.h"
)

# Put all libraries into a variable
set(LIBS

	GLFW
	Glad
	ImGui
	libnanovdb
)

set(INCLUDES
	"vendor/GLFW/include"
	"vendor/Glad/include"
	"vendor/imgui"
	"vendor/glm"
	"vendor/spdlog/include"
	"vendor/nlohmann/include"
	"vendor/fx"
	"vendor/stb"
	"src"
	${NANOVDB_HEADERS}
)

add_compile_definitions(GLFW_INCLUDE_NONE)

# Define the include DIRs
target_include_directories(Lumen PUBLIC

	${INCLUDES}
)

# Define the link libraries
target_link_libraries(Lumen 

	${LIBS}
)