cmake_minimum_required (VERSION 3.16)
enable_language(CUDA)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

#Add CUDA files
file(GLOB_RECURSE CUDA_FILES 

	${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

#generate .ptx files from .cu files
add_library(CudaPTX OBJECT ${CUDA_FILES})
set_property(TARGET CudaPTX PROPERTY CUDA_PTX_COMPILATION ON)

target_include_directories(CudaPTX PUBLIC
	"vendor/Include"
	"vendor/Include/Cuda"
)

install(TARGETS CudaPTX
   OBJECTS DESTINATION bin/ptx 
)

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
	${CMAKE_CURRENT_SOURCE_DIR}/cmake/LumenConfig.h.in 
	${CMAKE_CURRENT_BINARY_DIR}/config/LumenConfig.h
)