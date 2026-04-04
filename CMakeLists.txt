cmake_minimum_required(VERSION 3.10)
project(LinearRegressionParallel)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
find_package(Catch2 3 REQUIRED)

set(IMPL "" CACHE STRING "Path to the implementation directory")

if(IMPL STREQUAL "")
    message(FATAL_ERROR "You must provide an implementation directory!")
endif()

file(GLOB_RECURSE IMPL_SOURCES "${IMPL}/*.cpp")
set(SOURCES ${IMPL_SOURCES})

get_filename_component(BINARY_NAME ${IMPL} NAME)

add_executable(${BINARY_NAME} ${SOURCES})

target_include_directories(${BINARY_NAME} PRIVATE ${IMPL})

if(NOT MSVC)
    target_compile_options(${BINARY_NAME} PRIVATE
        $<$<CONFIG:Release>:-O3 -march=native -ffast-math>
    )

    target_compile_options(${BINARY_NAME} PRIVATE
        $<$<CONFIG:Debug>:-O0 -g -Wall -Wextra>
    )
endif()

target_link_libraries(${BINARY_NAME} PRIVATE OpenMP::OpenMP_CXX)

message(STATUS "Configuring build for: ${BINARY_NAME}")
