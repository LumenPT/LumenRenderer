cmake_minimum_required (VERSION 3.16)
enable_language(CUDA)

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

# Add source to this project's executable
add_executable (Sandbox  ${HEADER_FILES} ${SOURCE_FILES})

include(${CMAKE_SOURCE_DIR}/cmake/functions.cmake)
assign_source_group(${HEADER_FILES})
assign_source_group(${SOURCE_FILES})

set_property(TARGET Sandbox PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
set_property(TARGET Sandbox APPEND PROPERTY COMPILE_DEFINITIONS GLFW_INCLUDE_NONE)

if(${USE_WAVEFRONT})
	set_property(TARGET Sandbox APPEND PROPERTY COMPILE_DEFINITIONS WAVEFRONT)
endif()

#Add header directory includes
target_include_directories(Sandbox PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}/../Lumen/src"
	"${CMAKE_CURRENT_SOURCE_DIR}/../Lumen/vendor"
	"${CMAKE_CURRENT_SOURCE_DIR}/../Lumen/vendor/glm"
	"${CMAKE_CURRENT_SOURCE_DIR}/../Lumen/vendor/spdlog/include"

	"${CMAKE_CURRENT_SOURCE_DIR}/../LumenPT/vendor/Include"
	"${CMAKE_CURRENT_SOURCE_DIR}/../LumenPT/vendor/Include/Cuda"
	"${CMAKE_CURRENT_SOURCE_DIR}/../LumenPT/src"
)

target_link_libraries(Sandbox
	Lumen
	LumenPT
)

add_custom_command(TARGET Sandbox POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/../LumenPT/vendor/Dlls"
        $<TARGET_FILE_DIR:Sandbox>)