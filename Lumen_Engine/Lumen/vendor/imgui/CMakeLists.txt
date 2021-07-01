cmake_minimum_required (VERSION 3.16)
project(ImGui)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#Add header files
file(GLOB_RECURSE HEADER_FILES 

	"imconfig.h"
    "imgui.h"
    "imgui_internal.h"
    "imstb_rectpack.h"
    "imstb_textedit.h"
    "imstb_truetype.h"
    "implot.h"
    "implot_internal.h"
)

#Add source files
file(GLOB_RECURSE SOURCE_FILES 

	"imgui.cpp"
    "imgui_draw.cpp"
    "imgui_tables.cpp"
    "imgui_widgets.cpp"
    "imgui_demo.cpp"    
    "implot.cpp"
    "implot_items.cpp"
    "implot_demo.cpp"
)

# Add header and source files to library
add_library (${PROJECT_NAME} STATIC ${HEADER_FILES} ${SOURCE_FILES})

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
