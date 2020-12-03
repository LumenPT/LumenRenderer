cmake_minimum_required (VERSION 3.16)
project(LumenPT)

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
add_library (${PROJECT_NAME} STATIC ${HEADER_FILES} ${SOURCE_FILES})

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
)

# Define the include DIRs
target_include_directories(${PROJECT_NAME} PUBLIC
	${INCLUDES}
)

# Define the link libraries
target_link_libraries(${PROJECT_NAME} 
	${LIBS}
)