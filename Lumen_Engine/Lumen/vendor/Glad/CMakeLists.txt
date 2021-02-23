cmake_minimum_required (VERSION 3.16)
project(Glad)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#Add header files
file(GLOB_RECURSE HEADER_FILES 

	"include/glad/glad.h"
    "include/KHR/khrplatform.h"
)

#Add source files
file(GLOB_RECURSE SOURCE_FILES 

	"src/glad.c"
)

# Add header and source files to library
add_library (${PROJECT_NAME} STATIC ${HEADER_FILES} ${SOURCE_FILES})

# Put all libraries into a variable
set(LIBS
	
)

#Put all include directories into a variable
set(INCLUDES
	"include"
)

# Define the include DIRs
target_include_directories(${PROJECT_NAME} PUBLIC

	${INCLUDES}
)

# Define the link libraries
target_link_libraries(${PROJECT_NAME} 

	${LIBS}
)