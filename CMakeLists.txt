cmake_minimum_required(VERSION 3.10)
project(LinearRegressionParallel)

# 1. Setup C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. Find Dependencies
find_package(OpenMP REQUIRED)
find_package(Catch2 3 REQUIRED)

# 3. Define the Implementation Directory argument
# Example: cmake -B build -DIMPL=src/impl_parallel_nan
set(IMPL "" CACHE STRING "Path to the implementation directory (e.g., src/impl_nan)")

if(IMPL STREQUAL "")
    message(FATAL_ERROR "You must provide an implementation directory! \nUsage: cmake -B build -DIMPL=src/path_to_folder")
endif()

# 4. Automatically find all .cpp files in the chosen IMPL folder
file(GLOB_RECURSE IMPL_SOURCES "${IMPL}/*.cpp")

# 5. Combine source files
set(SOURCES ${IMPL_SOURCES})

# 6. Set Include Directories
include_directories(${IMPL})

# 7. Create the Executable
# We name the output binary after the last part of the folder path
get_filename_component(BINARY_NAME ${IMPL} NAME)
add_executable(${BINARY_NAME} ${SOURCES})

# 8. Link Libraries
target_link_libraries(${BINARY_NAME} PRIVATE OpenMP::OpenMP_CXX)

# 9. Setup Tests (pointing to the same implementation)
#file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp")
#add_executable(run_tests ${TEST_SOURCES} ${SOURCES})
#target_link_libraries(run_tests PRIVATE Catch2::Catch2WithMain OpenMP::OpenMP_CXX)

message(STATUS "Configuring build for: ${BINARY_NAME}")
