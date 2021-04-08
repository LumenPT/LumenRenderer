cmake_minimum_required(VERSION 3.16)
enable_language(CUDA)

#specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

#specify the CUDA standard
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED true)

#Add header files
file(GLOB_RECURSE CPP_HEADER_FILES 

	${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp
)

#Add source files
file(GLOB_RECURSE CPP_SOURCE_FILES 

	${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

if((NOT CPP_HEADER_FILES) AND (NOT CPP_SOURCE_FILES))
    message("No files found for project Tests!")
    return()
endif()

add_executable(Tests ${CPP_HEADER_FILES} ${CPP_SOURCE_FILES})

include(${CMAKE_SOURCE_DIR}/cmake/functions.cmake)
assign_source_group(${CPP_HEADER_FILES})
assign_source_group(${CPP_SOURCE_FILES})

set_property(TARGET Tests PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
set_property(TARGET Tests APPEND PROPERTY COMPILE_DEFINITIONS GLFW_INCLUDE_NONE)

if(${USE_WAVEFRONT})
	set_property(TARGET Tests APPEND PROPERTY COMPILE_DEFINITIONS WAVEFRONT)
endif()

target_include_directories(Tests PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}/../Lumen/src"
	"${CMAKE_CURRENT_SOURCE_DIR}/../Lumen/vendor"
	"${CMAKE_CURRENT_SOURCE_DIR}/../Lumen/vendor/glm"
	"${CMAKE_CURRENT_SOURCE_DIR}/../Lumen/vendor/spdlog/include"

	"${CMAKE_CURRENT_SOURCE_DIR}/../LumenPT/vendor/Include"
	"${CMAKE_CURRENT_SOURCE_DIR}/../LumenPT/vendor/Include/Cuda"
	"${CMAKE_CURRENT_SOURCE_DIR}/../LumenPT/src"
)

target_link_libraries(Tests
	Lumen
	LumenPT
)

add_custom_command(TARGET Tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/../LumenPT/vendor/Dlls"
        $<TARGET_FILE_DIR:Tests>)