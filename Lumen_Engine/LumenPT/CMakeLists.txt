cmake_minimum_required (VERSION 3.20)
enable_language(CUDA)

set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED True)



#CudaPTX project ---------------------------------------------------------------------------------------

#Add CUDA files
file(GLOB_RECURSE CUDA_HEADER_FILES
	${CMAKE_CURRENT_SOURCE_DIR}/src/Shaders/*.cuh
)

file(GLOB_RECURSE CUDA_SOURCE_FILES 
	${CMAKE_CURRENT_SOURCE_DIR}/src/Shaders/*.cu
)

#generate .ptx files from .cu files
add_library(CUDAShaders OBJECT ${CUDA_HEADER_FILES} ${CUDA_SOURCE_FILES})
set_property(TARGET CUDAShaders PROPERTY CUDA_PTX_COMPILATION ON)
set_property(TARGET CUDAShaders PROPERTY CUDA_ARCHITECTURES 52-real)

target_include_directories(CUDAShaders PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/Include"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/Include/Cuda"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/Shaders"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/Shaders/CppCommon"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/openvdb/nanovdb"
	${NANOVDB_HEADERS}
)

set(CUDA_FILES_OUTPUT "")
cmake_path(SET CUDA_FILES_OUTPUT_DIR NORMALIZE ${CMAKE_CURRENT_SOURCE_DIR}/../Sandbox/Assets/Shaders)
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
	COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_OBJECTS:CUDAShaders> ${CUDA_FILES_OUTPUT_DIR}
	#COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_OBJECTS:CudaPTX> ${CUDA_FILES_OUTPUT_DIR}
	DEPENDS CUDAShaders
	VERBATIM
	COMMAND_EXPAND_LISTS)

add_custom_target(CopyCUDAShaders ALL DEPENDS ${CUDA_FILES_OUTPUT})



#CudaKernels project ---------------------------------------------------------------------------------------

file(GLOB_RECURSE CUDA_HEADER_FILES

	${CMAKE_CURRENT_SOURCE_DIR}/src/CUDAKernels/*.cuh
	${CMAKE_CURRENT_SOURCE_DIR}/src/Tools/*.cuh
)

file(GLOB_RECURSE CUDA_SOURCE_FILES
	${CMAKE_CURRENT_SOURCE_DIR}/src/CUDAKernels/*.cu
	${CMAKE_CURRENT_SOURCE_DIR}/src/Tools/*.cu
)

set(CUDAKernels_INCLUDES
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/Include"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/Include/Cuda"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/Include/sutil"
	"${CMAKE_CURRENT_SOURCE_DIR}/../Lumen/vendor/glm"
	"${CMAKE_CURRENT_SOURCE_DIR}/../Lumen/src"
	"${CMAKE_CURRENT_SOURCE_DIR}/../Lumen/src/Lumen"
	"${CMAKE_CURRENT_SOURCE_DIR}/../Lumen/src/Lumen/AssetLoading"
	"${CMAKE_CURRENT_SOURCE_DIR}/../Lumen/src/Lumen/ModelLoading"
	"${CMAKE_CURRENT_BINARY_DIR}/config"
	${NANOVDB_HEADERS}
)

set(CUDAKernels_LIBS libnanovdb)

add_library(CUDAKernels STATIC ${CUDA_HEADER_FILES} ${CUDA_SOURCE_FILES})
set_property(TARGET CUDAKernels PROPERTY CUDA_ARCHITECTURES 52-real)
set_property(TARGET CUDAKernels PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET CUDAKernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(CUDAKernels PUBLIC ${CUDAKernels_INCLUDES})

target_link_libraries(CUDAKernels ${CUDAKernels_LIBS})

assign_source_group(${CUDA_HEADER_FILES})
assign_source_group(${CUDA_SOURCE_FILES})



#LUMENPT project ---------------------------------------------------------------------------------------

#Add header files
file(GLOB_RECURSE CPP_HEADER_FILES 

	${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.inl
)

if(${USE_NVIDIA_DENOISER})
	list(APPEND CPP_HEADER_FILES 
	"vendor/RayTracingDenoiser/LumenPT/DX12Wrapper/NRIDX12Wrapper.h"
	"vendor/RayTracingDenoiser/LumenPT/DX12Wrapper/Helper.h"
	"vendor/RayTracingDenoiser/LumenPT/DX12Wrapper/Utils.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/RayTracingDenoiser/_NRD_SDK/Integration/NRDIntegration.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/RayTracingDenoiser/_NRD_SDK/Integration/NRDIntegration.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/RayTracingDenoiser/External/MathLib/MathLib.h"

	)
endif()

#Add source files
file(GLOB_RECURSE CPP_SOURCE_FILES 

	${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

if(${USE_NVIDIA_DENOISER})
	list(APPEND CPP_SOURCE_FILES
	"vendor/RayTracingDenoiser/LumenPT/DX12Wrapper/NRIDX12Wrapper.cpp"
	"vendor/RayTracingDenoiser/LumenPT/DX12Wrapper/Utils.cpp"
	#"vendor/RayTracingDenoiser/NRD/Integration/NRDIntegration.cpp"
	)
endif()

set(LUMENPT_INCLUDES
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/Include"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/Include/Cuda"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/openvdb/openvdb"
	"${CMAKE_CURRENT_SOURCE_DIR}/../Lumen/src"
	"${CMAKE_CURRENT_SOURCE_DIR}/../Lumen/src/Lumen"
	"${CMAKE_CURRENT_SOURCE_DIR}/../Lumen/src/AssetLoading"
	"${CMAKE_CURRENT_SOURCE_DIR}/../Lumen/vendor/spdlog/include/"
	"${CMAKE_CURRENT_SOURCE_DIR}/../Lumen/vendor/glm"
	"${CMAKE_CURRENT_BINARY_DIR}/config"
	${NANOVDB_HEADERS}
)

if(${USE_NVIDIA_DENOISER})
	list(APPEND LUMENPT_INCLUDES
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/RayTracingDenoiser/LumenPT" 
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/RayTracingDenoiser/_NRD_SDK/Include"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/RayTracingDenoiser/_NRD_SDK/Integration"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/RayTracingDenoiser/_NRI_SDK/Include"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/RayTracingDenoiser/External"
	#"${CMAKE_CURRENT_SOURCE_DIR}/vendor/RayTracingDenoiser/NRI/Include"
	#"${CMAKE_CURRENT_SOURCE_DIR}/vendor/RayTracingDenoiser/NRD/Integration" 
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/Include/Detex")
endif()

# Put all libraries into a variable
set(LUMENPT_LIBS

	libnanovdb
	openvdb_static
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/Libs/glad.lib"
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

if(${USE_NVIDIA_DENOISER})
	list(APPEND LUMENPT_LIBS 
	
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/RayTracingDenoiser/_NRD_SDK/Lib/Release/NRD.lib"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/RayTracingDenoiser/_NRI_SDK/Lib/Release/NRI.lib"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/RayTracingDenoiser/_NRI_SDK/Lib/Release/NRI_Creation.lib"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/RayTracingDenoiser/_NRI_SDK/Lib/Release/NRI_Validation.lib"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/RayTracingDenoiser/_NRI_SDK/Lib/Release/NRI_D3D11.lib"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/RayTracingDenoiser/_NRI_SDK/Lib/Release/NRI_D3D12.lib"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/RayTracingDenoiser/_NRI_SDK/Lib/Release/NRI_VK.lib"

	#NRD NRI NRI_D3D12 NRI_Validation
	)
endif()

# Add header and source files to library
add_library (LumenPT STATIC ${CPP_HEADER_FILES} ${CPP_SOURCE_FILES})
add_dependencies(LumenPT CUDAKernels)
add_dependencies(LumenPT CopyCUDAShaders)
set_property(TARGET LumenPT PROPERTY CUDA_ARCHITECTURES 52-real)
set_property(TARGET LumenPT APPEND PROPERTY COMPILE_DEFINITIONS LUMEN_PT)
set_property(TARGET LumenPT APPEND PROPERTY COMPILE_DEFINITIONS $<$<CONFIG:Debug>:CUDA_CHECK OPTIX_CHECK>)

if(${USE_WAVEFRONT})
	set_property(TARGET LumenPT APPEND PROPERTY COMPILE_DEFINITIONS WAVEFRONT)
endif()

assign_source_group(${CPP_HEADER_FILES})
assign_source_group(${CPP_SOURCE_FILES})

# Define the include DIRs
target_include_directories(LumenPT PUBLIC ${LUMENPT_INCLUDES})

# Define the link libraries
target_link_libraries(LumenPT ${LUMENPT_LIBS} CUDAKernels)

# Config file where CMake will specify the location of the project directory
# configure_file(
# 	${CMAKE_CURRENT_SOURCE_DIR}/cmake/LumenPTConfig.h.in 
# 	${CMAKE_CURRENT_BINARY_DIR}/config/LumenPTConfig.h
# )