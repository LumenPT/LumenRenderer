cmake_minimum_required (VERSION 3.16)
project(Sandbox)

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

# Add source to this project's executable
add_executable (${PROJECT_NAME}  ${HEADER_FILES} ${SOURCE_FILES})

set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT "Sandbox")

#Add header directory includes
target_include_directories(${PROJECT_NAME} PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}/../Lumen/src"
	"${CMAKE_CURRENT_SOURCE_DIR}/../Lumen/vendor"
	"${CMAKE_CURRENT_SOURCE_DIR}/../Lumen/vendor/glm"
	"${CMAKE_CURRENT_SOURCE_DIR}/../Lumen/vendor/spdlog/include"

	"${CMAKE_CURRENT_SOURCE_DIR}/../LumenPT/vendor/Include"
	"${CMAKE_CURRENT_SOURCE_DIR}/../LumenPT/vendor/Include/Cuda"
	"${CMAKE_CURRENT_SOURCE_DIR}/../LumenPT/src"
)

target_link_libraries(${PROJECT_NAME} 
	Lumen
	LumenPT
)