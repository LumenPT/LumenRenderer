cmake_minimum_required (VERSION 3.16)
project(GLFW)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#Add header files
file(GLOB_RECURSE HEADER_FILES 

	"include/GLFW/glfw3.h"
    "include/GLFW/glfw3native.h"
    "src/glfw_config.h"
)

#Add source files
file(GLOB_RECURSE SOURCE_FILES 
	"src/context.c"
    "src/init.c"
    "src/input.c"
    "src/monitor.c"
    "src/vulkan.c"
    "src/window.c"
    "src/win32_init.c"
    "src/win32_joystick.c"
    "src/win32_monitor.c"
    "src/win32_time.c"
    "src/win32_thread.c"
    "src/win32_window.c"
    "src/wgl_context.c"
    "src/egl_context.c"
    "src/osmesa_context.c"
)

# Add header and source files to library
add_library (${PROJECT_NAME} STATIC ${HEADER_FILES} ${SOURCE_FILES})

add_compile_definitions(_GLFW_WIN32)
add_compile_definitions(CRT_SECURE_NO_WARNINGS)

# Put all libraries into a variable
set(LIBS
	
)

#Put all include directories into a variable
set(INCLUDES
	
)

# Define the include DIRs
target_include_directories(${PROJECT_NAME} PUBLIC

	${INCLUDES}
)

# Define the link libraries
target_link_libraries(${PROJECT_NAME} 

	${LIBS}
)