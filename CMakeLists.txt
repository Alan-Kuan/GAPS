cmake_minimum_required(VERSION 3.25.2)
project(shoz LANGUAGES C CXX CUDA)

option(BUILD_STATIC_LIBS "Build the static library" ON)  # zenoh-pico's option
option(BUILD_SHARED_LIBS "Build the shared library" ON)  # zenoh-pico's option
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TORCH_SUPPORT "Build a Python module with PyTorch support" OFF)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
endif()

# Dependencies
find_package(CUDAToolkit REQUIRED)
if(BUILD_TORCH_SUPPORT)
    find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)

    execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
    find_package(nanobind CONFIG REQUIRED)
endif()

include(FetchContent)

FetchContent_declare(zenohpico GIT_REPOSITORY "https://github.com/eclipse-zenoh/zenoh-pico" GIT_TAG 0.11.0)
FetchContent_MakeAvailable(zenohpico)
FetchContent_declare(zenohcpp GIT_REPOSITORY "https://github.com/eclipse-zenoh/zenoh-cpp" GIT_TAG 0.11.0)
FetchContent_MakeAvailable(zenohcpp)

# Utility Function
function(shoz_add_target targetName)
    set(flags IS_EXE PIC)
    set(listArgs SRCS INCS LIBS)
    cmake_parse_arguments(ARG "${flags}" "" "${listArgs}" ${ARGN})

    if (NOT ARG_SRCS)
        message(FATAL_ERROR "[shoz_add_target]: missing required argument SRCS")
    endif()

    if (ARG_IS_EXE)
        add_executable(${targetName})
    else()
        add_library(${targetName} STATIC)
    endif()

    target_sources(${targetName} PRIVATE ${ARG_SRCS})

    if (ARG_INCS)
        target_include_directories(${targetName} PRIVATE ${ARG_INCS})
    endif()
    if (ARG_LIBS)
        target_link_libraries(${targetName} PRIVATE ${ARG_LIBS})
    endif()

    if (ARG_PIC)
        set_target_properties(${targetName} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()
endfunction()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS})
add_subdirectory(src)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()