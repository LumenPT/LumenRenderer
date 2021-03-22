cmake_minimum_required (VERSION 3.16)
enable_language(CUDA)

set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

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

#Add CUDA files
file(GLOB_RECURSE CUDA_HEADER_FILES
	${CMAKE_CURRENT_SOURCE_DIR}/src/Shaders/*.cuh
)

file(GLOB_RECURSE CUDA_SOURCE_FILES 
	${CMAKE_CURRENT_SOURCE_DIR}/src/Shaders/*.cu
)

#generate .ptx files from .cu files
add_library(CudaPTX OBJECT ${CUDA_HEADER_FILES} ${CUDA_SOURCE_FILES})
set_property(TARGET CudaPTX PROPERTY CUDA_PTX_COMPILATION ON)

target_include_directories(CudaPTX PUBLIC
	"vendor/Include"
	"vendor/Include/Cuda"
	"src/Shaders"
	"src/Shaders/CppCommon"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/openvdb/nanovdb"
	${NANOVDB_HEADERS}
)

set(CUDA_FILES_OUTPUT "")
set(CUDA_FILES_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/Assets/PrecompiledShaders)
#if(${CONFIG_NAME})
	message("ConfigName: ${CONFIG_NAME}")
	set(CUDA_FILES_OUTPUT_DIR ${CUDA_FILES_OUTPUT_DIR}/${CONFIG_NAME})
#endif()


#set(CUDA_FILES_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/Assets/PrecompiledShaders/${CMAKE_CFG_INTDIR})
#https://cmake.org/cmake/help/v3.19/variable/CMAKE_CFG_INTDIR.html#variable:CMAKE_CFG_INTDIR

#replace input file paths with output path and extension. Capture first part because otherwise first path will not be replaced.
include(${CMAKE_SOURCE_DIR}/cmake/functions.cmake)
changePath("(.*\\/)([A-Za-z0-9_\\.]+\\.)cu\\;?$" "${CUDA_FILES_OUTPUT_DIR}/\\2ptx" CUDA_FILES_OUTPUT "${CUDA_SOURCE_FILES}")

message("----------[COPIED CUDA FILES OUTPUT]----------")
printList(${CUDA_FILES_OUTPUT})
message("----------------------------------------------")

add_custom_command(
	OUTPUT ${CUDA_FILES_OUTPUT}
	COMMAND ${CMAKE_COMMAND} -E make_directory ${CUDA_FILES_OUTPUT_DIR}
	COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_OBJECTS:CudaPTX> ${CUDA_FILES_OUTPUT_DIR}
	#COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_OBJECTS:CudaPTX> ${CUDA_FILES_OUTPUT_DIR}
	DEPENDS CudaPTX
	VERBATIM
	COMMAND_EXPAND_LISTS)

add_custom_target(copyCUDAOutput ALL DEPENDS ${CUDA_FILES_OUTPUT})

#Add header files
file(GLOB_RECURSE CPP_HEADER_FILES 

	${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.inl
	${CMAKE_CURRENT_SOURCE_DIR}/src/CUDAKernels/*.cuh
)

#Add source files
file(GLOB_RECURSE CPP_SOURCE_FILES 

	${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/CUDAKernels/*.cu
)

# Add header and source files to library
add_library (LumenPT STATIC ${CPP_HEADER_FILES} ${CPP_SOURCE_FILES})
add_dependencies(LumenPT CudaPTX)
set_property(TARGET LumenPT APPEND_STRING PROPERTY COMPILE_DEFINITIONS CONFIG_NAME=\"${CONFIG_NAME}\")
set_property(TARGET LumenPT APPEND PROPERTY COMPILE_DEFINITIONS LUMEN_PT)

if(${USE_WAVEFRONT})
	set_property(TARGET LumenPT APPEND PROPERTY COMPILE_DEFINITIONS WAVEFRONT)
endif()

assign_source_group(${CPP_HEADER_FILES})
assign_source_group(${CPP_SOURCE_FILES})

# Put all libraries into a variable
set(LIBS
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/Libs/Cuda/cuda.lib"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/Libs/Cuda/cudart.lib"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/Libs/Cuda/cudadevrt.lib"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/Libs/glad.lib"
	libnanovdb
	openvdb_static

	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/Libs/tbb/tbb.lib"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/Libs/tbb/tbbmalloc.lib"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/Libs/tbb/tbbmalloc_proxy.lib"

	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/Libs/blosc.lib"

	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/Libs/zlib.lib"

	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/Libs/OpenEXR/Half-2_5.lib"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/Libs/OpenEXR/Iex-2_5.lib"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/Libs/OpenEXR/IexMath-2_5.lib"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/Libs/OpenEXR/IlmImf-2_5.lib"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/Libs/OpenEXR/IlmImfUtil-2_5.lib"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/Libs/OpenEXR/IlmThread-2_5.lib"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/Libs/OpenEXR/Imath-2_5.lib"


)

#include(${CMAKE_CURRENT_SOURCE_DIR}/vendor/openvdb/openvdb/openvdb/CMakeLists.txt)
#get_cmake_property(_variableNames OPENVDB_CORE_DEPENDENT_LIBS)
#message(test)
#message("${OPENVDB_CORE_DEPENDENT_LIBS}")

set(INCLUDES
	"vendor/Include"
	"vendor/Include/Cuda"
	"../Lumen/src/AssetLoading"
	"../Lumen/src/Lumen"
	"../Lumen/src"
	"${CMAKE_CURRENT_SOURCE_DIR}/../Lumen/vendor/glm"
	"${CMAKE_CURRENT_BINARY_DIR}/config"
	${NANOVDB_HEADERS}
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/openvdb/openvdb"
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