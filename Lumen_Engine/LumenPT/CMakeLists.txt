cmake_minimum_required (VERSION 3.20)
enable_language(CUDA)

#set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# Config file where CMake will specify the location of the project directory
configure_file(
	${CMAKE_CURRENT_SOURCE_DIR}/cmake/LumenPTConfig.h.in 
	${CMAKE_CURRENT_BINARY_DIR}/config/LumenPTConfig.h
)

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
set_property(TARGET CUDAShaders PROPERTY CUDA_ARCHITECTURES 60-real)

target_compile_options(CUDAShaders PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>: -G>)
target_compile_options(CUDAShaders PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>: -use_fast_math>)

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
	COMMAND if exist "${CUDA_FILES_OUTPUT_DIR}\\" 
			${CMAKE_COMMAND} -E rm -rf "${CUDA_FILES_OUTPUT_DIR}\\" &&
			${CMAKE_COMMAND} -E echo "[INFO]: Cleaned up shader directory: ${CUDA_FILES_OUTPUT_DIR}"
	COMMAND ${CMAKE_COMMAND} -E make_directory ${CUDA_FILES_OUTPUT_DIR}
	#COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_OBJECTS:CUDAShaders> ${CUDA_FILES_OUTPUT_DIR}
	COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_OBJECTS:CUDAShaders> ${CUDA_FILES_OUTPUT_DIR} &&
			${CMAKE_COMMAND} -E echo "[INFO]: Copied $<TARGET_OBJECTS:CUDAShaders> to ${CUDA_FILES_OUTPUT_DIR}"
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
	${NANOVDB_HEADERS}
)

set(CUDAKernels_LIBS libnanovdb)

add_library(CUDAKernels STATIC ${CUDA_HEADER_FILES} ${CUDA_SOURCE_FILES})
set_property(TARGET CUDAKernels PROPERTY CUDA_ARCHITECTURES 60-real)
set_property(TARGET CUDAKernels PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET CUDAKernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_compile_options(CUDAKernels PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>: -G>)
target_compile_options(CUDAKernels PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>: -use_fast_math>)

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

	${CMAKE_CURRENT_BINARY_DIR}/Config.h	

)

if(${USE_NVIDIA_DENOISER})
	list(APPEND CPP_HEADER_FILES 
	"vendor/RayTracingDenoiser/LumenPT/Nvidia/NRDWrapper.h"
	"vendor/RayTracingDenoiser/LumenPT/Nvidia/Helper.h"
	"vendor/RayTracingDenoiser/LumenPT/Nvidia/Utils.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/RayTracingDenoiser/_NRD_SDK/Integration/NRDIntegration.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/RayTracingDenoiser/_NRD_SDK/Integration/NRDIntegration.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/RayTracingDenoiser/External/MathLib/MathLib.h"

	)
endif()

if(${USE_NVIDIA_DLSS})
	list(APPEND CPP_HEADER_FILES 
	"vendor/DLSS_Private/LumenPT/Nvidia/DLSSWrapper.h"

	)
endif()

#Add source files
file(GLOB_RECURSE CPP_SOURCE_FILES 

	${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

if(${USE_NVIDIA_DENOISER})
	list(APPEND CPP_SOURCE_FILES
	"vendor/RayTracingDenoiser/LumenPT/Nvidia/NRDWrapper.cpp"
	"vendor/RayTracingDenoiser/LumenPT/Nvidia/Utils.cpp"
	#"vendor/RayTracingDenoiser/NRD/Integration/NRDIntegration.cpp"
	)
endif()

if(${USE_NVIDIA_DLSS})
	list(APPEND CPP_SOURCE_FILES
	"vendor/DLSS_Private/LumenPT/Nvidia/DLSSWrapper.cpp"

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
	"${CMAKE_CURRENT_SOURCE_DIR}/src/"
	#"${CMAKE_CURRENT_BINARY_DIR}/config"
	"${CMAKE_CURRENT_SOURCE_DIR}/../Lumen/vendor/fx"
	"${CMAKE_CURRENT_SOURCE_DIR}/../Lumen/vendor/nlohmann/include"
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

if(${USE_NVIDIA_DLSS})
	list(APPEND LUMENPT_INCLUDES
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/DLSS_Private/Include"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/DLSS_Private/LumenPT"
	)
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
	
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/RayTracingDenoiser/_NRD_SDK/Lib/Debug/NRD.lib"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/RayTracingDenoiser/_NRI_SDK/Lib/Debug/NRI.lib"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/RayTracingDenoiser/_NRI_SDK/Lib/Debug/NRI_Creation.lib"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/RayTracingDenoiser/_NRI_SDK/Lib/Debug/NRI_Validation.lib"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/RayTracingDenoiser/_NRI_SDK/Lib/Debug/NRI_D3D11.lib"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/RayTracingDenoiser/_NRI_SDK/Lib/Debug/NRI_D3D12.lib"
	"${CMAKE_CURRENT_SOURCE_DIR}/vendor/RayTracingDenoiser/_NRI_SDK/Lib/Debug/NRI_VK.lib"

	#NRD NRI NRI_D3D12 NRI_Validation
	)
endif()

if(${USE_NVIDIA_DLSS})
	file(GLOB_RECURSE DLSS_LIBS 

	#${CMAKE_CURRENT_SOURCE_DIR}/vendor/DLSS_Private/Lib/x64/*_d_dbg.lib #TODO: use release version in release mode
	#${CMAKE_CURRENT_SOURCE_DIR}/vendor/DLSS_Private/Lib/x64/nvsdk_ngx_d_dbg.lib #TODO: use release version in release mode
	${CMAKE_CURRENT_SOURCE_DIR}/vendor/DLSS_Private/Lib/x64/*_d.lib #TODO: use release version in release mode
	)

	list(APPEND LUMENPT_LIBS 
	
	${DLSS_LIBS}
	)
endif()

# Add header and source files to library
add_library (LumenPT STATIC ${CPP_HEADER_FILES} ${CPP_SOURCE_FILES})
add_dependencies(LumenPT CUDAKernels)
add_dependencies(LumenPT CopyCUDAShaders)
set_property(TARGET LumenPT PROPERTY CUDA_ARCHITECTURES 60-real)
set_property(TARGET LumenPT APPEND PROPERTY COMPILE_DEFINITIONS LUMEN_PT)
set_property(TARGET LumenPT APPEND PROPERTY COMPILE_DEFINITIONS $<$<CONFIG:Debug>:CUDA_CHECK OPTIX_CHECK OPTIX_DEBUG DX11_CHECK>)

target_compile_options(LumenPT PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>: -G>)
target_compile_options(LumenPT PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>: -use_fast_math>)

if(${USE_WAVEFRONT})
	set_property(TARGET LumenPT APPEND PROPERTY COMPILE_DEFINITIONS WAVEFRONT)
endif()

if(${USE_NVIDIA_DENOISER})
	set_property(TARGET LumenPT APPEND PROPERTY COMPILE_DEFINITIONS USE_NVIDIA_DENOISER)
endif()

if(${USE_NVIDIA_DLSS})
	set_property(TARGET LumenPT APPEND PROPERTY COMPILE_DEFINITIONS USE_NVIDIA_DLSS)
endif()


assign_source_group(${CPP_HEADER_FILES})
assign_source_group(${CPP_SOURCE_FILES})

# Define the include DIRs
target_include_directories(LumenPT PUBLIC ${LUMENPT_INCLUDES})

# Define the link librarie
target_link_libraries(LumenPT ${LUMENPT_LIBS} CUDAKernels)

# Config file where CMake will specify the location of the project directory
#configure_file(
#	${CMAKE_CURRENT_SOURCE_DIR}/cmake/LumenPTConfig.h.in 
#	${CMAKE_CURRENT_BINARY_DIR}/config/LumenPTConfig.h
#)