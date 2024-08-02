cmake_minimum_required(VERSION 3.25.2)
project(shoz LANGUAGES C CXX CUDA)

option(BUILD_STATIC_LIBS "Build the static library" ON)
option(BUILD_SHARED_LIBS "Build the shared library" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TESTS "Build testing and experiment programs" ON)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

include(FetchContent)

FetchContent_declare(zenohpico GIT_REPOSITORY "https://github.com/eclipse-zenoh/zenoh-pico" GIT_TAG 0.11.0.3)
FetchContent_MakeAvailable(zenohpico)
FetchContent_declare(zenohcpp GIT_REPOSITORY "https://github.com/eclipse-zenoh/zenoh-cpp" GIT_TAG 0.11.0.3)
FetchContent_MakeAvailable(zenohcpp)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS})
add_subdirectory(src)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()