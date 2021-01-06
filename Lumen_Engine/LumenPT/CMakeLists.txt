cmake_minimum_required (VERSION 3.16)
enable_language(CUDA)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED True)


#function to transform input paths to output paths.
function(changePath REGEX_MATCH REGEX_REPLACE RESULT_VAR_NAME)

	set(RESULT "")

	foreach(ARG IN LISTS ARGN)

		string(REGEX REPLACE ${REGEX_MATCH} ${REGEX_REPLACE} OUTPUTPATH ${ARG})
		list(APPEND RESULT ${OUTPUTPATH})

		#message("\n Replaced: ${ARG} \n with: ${OUTPUTPATH}")

	endforeach()

	set(${RESULT_VAR_NAME} ${RESULT} PARENT_SCOPE)
	
endfunction()



#Add CUDA files
file(GLOB_RECURSE CUDA_FILES 

	${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

#generate .ptx files from .cu files
add_library(CudaPTX OBJECT ${CUDA_FILES})
set_property(TARGET CudaPTX PROPERTY CUDA_PTX_COMPILATION ON)
#set_property(TARGET CudaPTX PROPERTY CUDA_ARCHITECTURES 52)

target_include_directories(CudaPTX PUBLIC
	"vendor/Include"
	"vendor/Include/Cuda"
)

set(CUDA_FILES_OUTPUT "")
set(CUDA_FILES_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/Assets/PrecompiledShaders)

if(	${CMAKE_GENERATOR} STREQUAL "Visual Studio 8 2005" OR
	${CMAKE_GENERATOR} STREQUAL "Visual Studio 9 2008")

	set(CUDA_FILES_OUTPUT_DIR ${CUDA_FILES_OUTPUT_DIR}/$(ConfigurationName))

elseif(
	${CMAKE_GENERATOR} STREQUAL "Visual Studio 10 2010" OR
	${CMAKE_GENERATOR} STREQUAL "Visual Studio 11 2012" OR
	${CMAKE_GENERATOR} STREQUAL "Visual Studio 12 2013" OR
	${CMAKE_GENERATOR} STREQUAL "Visual Studio 14 2015" OR
	${CMAKE_GENERATOR} STREQUAL "Visual Studio 15 2017" OR
	${CMAKE_GENERATOR} STREQUAL "Visual Studio 16 2019")

	set(CUDA_FILES_OUTPUT_DIR ${CUDA_FILES_OUTPUT_DIR}/$(Configuration))

elseif(
	${CMAKE_GENERATOR} STREQUAL "Xcode" OR
	${CMAKE_GENERATOR} STREQUAL "Ninja Multi-Config")
	set(CUDA_FILES_OUTPUT_DIR ${CUDA_FILES_OUTPUT_DIR}/$(CONFIGURATION))
endif()

#set(CUDA_FILES_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/Assets/PrecompiledShaders/${CMAKE_CFG_INTDIR})
#https://cmake.org/cmake/help/v3.19/variable/CMAKE_CFG_INTDIR.html#variable:CMAKE_CFG_INTDIR

#replace input file paths with output path and extension. Capture first part because otherwise first path will not be replaced.
changePath("(.*\\/)([A-Za-z0-9_\\.]+\\.)cu\\;?$" "${CUDA_FILES_OUTPUT_DIR}/\\2ptx" CUDA_FILES_OUTPUT "${CUDA_FILES}")

message("CUDA_FILES_OUTPUT: ${CUDA_FILES_OUTPUT}")
add_custom_command(
	OUTPUT ${CUDA_FILES_OUTPUT}
	COMMAND ${CMAKE_COMMAND} -E make_directory ${CUDA_FILES_OUTPUT_DIR}
	COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_OBJECTS:CudaPTX> ${CUDA_FILES_OUTPUT_DIR}
	#COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_OBJECTS:CudaPTX> ${CUDA_FILES_OUTPUT_DIR}/$<CONFIG>
	DEPENDS CudaPTX
	VERBATIM
	COMMAND_EXPAND_LISTS)

add_custom_target(copyCUDAOutput ALL DEPENDS ${CUDA_FILES_OUTPUT})

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
add_library (LumenPT STATIC ${HEADER_FILES} ${SOURCE_FILES})
add_dependencies(LumenPT CudaPTX)

include(${CMAKE_SOURCE_DIR}/cmake/functions.cmake)
assign_source_group(${HEADER_FILES})
assign_source_group(${SOURCE_FILES})

# Put all libraries into a variable
set(LIBS
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/Libs/Cuda/cuda.lib"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/Libs/Cuda/cudart.lib"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/Libs/Cuda/cudadevrt.lib"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/Libs/glad.lib"
)

set(INCLUDES
	"vendor/Include"
	"vendor/Include/Cuda"
	"../Lumen/src/AssetLoading"
	"../Lumen/src/Lumen"
	"../Lumen/src"
	"${CMAKE_CURRENT_SOURCE_DIR}/../Lumen/vendor/glm"
	"${CMAKE_CURRENT_BINARY_DIR}/config"
)

# Define the include DIRs
target_include_directories(LumenPT PUBLIC
	${INCLUDES}
)

# Define the link libraries
target_link_libraries(LumenPT 
	${LIBS}
)

# Config file where CMake will specify the location of the project directory
configure_file(
	${CMAKE_CURRENT_SOURCE_DIR}/cmake/LumenPTConfig.h.in 
	${CMAKE_CURRENT_BINARY_DIR}/config/LumenPTConfig.h
)